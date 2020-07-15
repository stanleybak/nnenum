'''
LP star implementation

Stanley Bak
'''

import numpy as np

from nnenum.lpinstance import LpInstance
from nnenum.util import Freezable
from nnenum.timerutil import Timers
from nnenum import kamenev

class LpStar(Freezable):
    '''generalized star set with lp constraints

    this is set up to efficiently encode: 
    linear transformation (using a_mat, a_rhs),
    constraints on initial variables (using csr / rhs)
    '''

    def __init__(self, a_mat, bias, box_bounds=None):
        assert a_mat is None or isinstance(a_mat, np.ndarray)
        assert bias is None or isinstance(bias, np.ndarray)
        
        self.a_mat = a_mat
        self.bias = bias

        # initialize lpi with no constraints
        self.lpi = None

        # for finding concrete counterexamples
        self.init_bm = None
        self.init_bias = None

        # cached lp result
        self.last_lp_result = None

        self.num_lps = 0 # stat, number of lps solved

        # box_bounds may be None if we're initializing in lp_star.copy()
        if box_bounds is not None:
            self.lpi = LpInstance()

            for i, (lb, ub) in enumerate(box_bounds):
                self.lpi.add_double_bounded_cols([f"i{i}"], lb, ub)

            # assign init bm and init bias as well in this case
            if a_mat is None:
                self.init_bm = None
            else:
                self.init_bm = self.a_mat.copy()

            if bias is None:
                self.init_bias = None
            else:
                self.init_bias = self.bias.copy()

        self.freeze_attrs()

    def __str__(self):
        rv = "LpStar with a_mat:\n"

        for i, row in enumerate(self.a_mat):
            rv += f"a_mat row {i}: {row}\n"

        rv += "bias:\n"

        for i, val in enumerate(self.bias):
            rv += f"bias {i}: {val}\n"

        rv += "and LP:\n"
        rv += str(self.lpi)

        return rv

    def copy(self):
        'return a copy of this lpstar'

        rv = LpStar(a_mat=self.a_mat.copy(), bias=self.bias.copy())

        rv.lpi = LpInstance(self.lpi)

        Timers.tic('copy init bm bias')

        if self.init_bm is None:
            rv.init_bm = None
        else:
            rv.init_bm = self.init_bm.copy()

        if self.init_bias is None:
            rv.init_bias = None
        else:
            rv.init_bias = self.init_bias.copy()
        Timers.toc('copy init bm bias')

        return rv

    def to_full_input(self, compressed_input):
        'convert possibly compressed input to full input'

        if self.init_bm is None:
            rv = compressed_input.copy()
        else:
            rv = np.dot(self.init_bm, compressed_input)

        if self.init_bias is not None:
            rv += self.init_bias

        if rv.dtype != self.a_mat.dtype:
            rv = rv.astype(self.a_mat.dtype)

        return rv

    def input_box_bounds(self, cur_box, max_dim=np.inf, count_lps=True):
        '''get input box bounds on this set, compared with the current bounds using lp

        returns a list of 3-tuples for each of the bounds that was adjusted:
        (dim, lb, ub)
        '''

        Timers.tic("star.input_box_bounds")

        dims = min(max_dim, self.lpi.get_num_cols())
        rv = []

        should_skip = np.zeros((dims, 2), dtype=bool)
        tol = 1e-8

        if cur_box is not None:
            #vec = np.ones(dims)

            while True:
                vec = np.array([1 if not should_skip[i, 0] else 0 for i in range(dims)], dtype=float)
                Timers.tic('lpi.minimize pre')
                res = self.lpi.minimize(vec)
                Timers.toc('lpi.minimize pre')

                skipped_all = True
                skipped_some = False
                
                for dim in range(dims):
                    if should_skip[dim, 0]:
                        continue
                    
                    if abs(res[dim] - cur_box[dim][0]) < tol:
                        should_skip[dim, 0] = True
                        skipped_some = True
                    else:
                        skipped_all = False

                if skipped_all or not skipped_some:
                    break

            # other side
            while True:
                vec = np.array([-1 if not should_skip[i, 1] else 0 for i in range(dims)], dtype=float)
                Timers.tic('lpi.minimize pre')
                res = self.lpi.minimize(vec)
                Timers.toc('lpi.minimize pre')

                skipped_all = True
                skipped_some = False

                for dim in range(dims):
                    if should_skip[dim, 1]:
                        continue
                    
                    if abs(res[dim] - cur_box[dim][1]) < tol:
                        should_skip[dim, 1] = True
                        skipped_some = True
                    else:
                        skipped_all = False

                if skipped_all or not skipped_some:
                    break

        for dim in range(dims):
            
            # adjust lb
            if not should_skip[dim, 0]:
                vec = np.array([1 if i == dim else 0 for i in range(dims)], dtype=float)
                Timers.tic('lpi.minimize post')
                res = self.lpi.minimize(vec)
                Timers.toc('lpi.minimize post')

                if count_lps:
                    self.num_lps += 1

                if cur_box is None or abs(res[dim] - cur_box[dim][0]) >= tol:
                    assert not should_skip[dim, 0]
                    rv.append([dim, res[dim], np.inf])

            # adjust ub
            if not should_skip[dim, 1]:
                vec = np.array([-1 if i == dim else 0 for i in range(dims)], dtype=float)

                Timers.tic('lpi.minimize post')
                res = self.lpi.minimize(vec)
                Timers.toc('lpi.minimize post')

                if count_lps:
                    self.num_lps += 1

                if cur_box is None or abs(res[dim] - cur_box[dim][1]) >= tol:
                    assert not should_skip[dim, 1]

                    if not rv or rv[-1][0] != dim:
                        rv.append([dim, -np.inf, res[dim]])
                    else:
                        rv[-1][2] = res[dim]

        Timers.toc("star.input_box_bounds")

        return rv

    def minimize_output(self, output_index, maximize=False):
        '''
        get the output value when one of the outputs is minimized (or maximized)

        if stop_at_zero is set, this will terminate the search once zero is crossed

        if you want the (input, output) pair to produce this output, use consutrct_last_io()
        '''

        Timers.tic('minimize_output')

        if self.a_mat.size == 0:
            value = self.bias
        else:
            row = self.a_mat[output_index]

            if maximize:
                row = -1 * row

            self.last_lp_result = lp_result = self.lpi.minimize(row)
            self.num_lps += 1

            num_init_vars = self.a_mat.shape[1]
            assert len(lp_result) == num_init_vars

            # single row
            value = self.a_mat[output_index].dot(lp_result) + self.bias[output_index]

        Timers.toc('minimize_output')

        return value

    def construct_last_io(self):
        '''construct the last concrete input/output pair from the optimization performed when minimize_output was called

        note that the input will be the compressed input if input space is not full dimensional
        '''

        Timers.tic('construct_last_io')
        
        i = self.last_lp_result
        
        o = np.dot(self.a_mat, i) + self.bias

        Timers.toc('construct_last_io')

        return [i, o]

    def minimize_vec(self, vec, return_io=False):
        '''optimize over this set

        vec is the vector of outputs we're optimizing over, None means use zero vector

        if return_io is true, returns a tuple (input, output); otherwise just output
        note that the cinput will be the compressed input if input space is not full dimensional

        returns all the outputs (coutput) if return_io=False, else (cinput, coutput)
        '''

        Timers.tic('star.minimize_vec')

        dtype = float if self.a_mat is None else self.a_mat.dtype

        if self.a_mat.size == 0:
            rv = self.bias
            self.last_lp_result = lp_result = np.array([], dtype=dtype)
        else:
            assert len(self.a_mat.shape) == 2, f"a_mat shape was {self.a_mat.shape}"

            if vec is None:
                vec = np.zeros((self.a_mat.shape[0],), dtype=dtype)

            assert len(vec) == self.a_mat.shape[0], f"minimize called with vector with {len(vec)} elements, " + \
                f"but set has {self.a_mat.shape[0]} outputs"

            #Timers.tic('setup')
            assert isinstance(vec, np.ndarray)

            lp_vec = np.dot(self.a_mat.T, vec)
            #lp_vec = vec.T.dot(self.a_mat).T

            num_init_vars = self.a_mat.shape[1]
            lp_vec.shape = (len(lp_vec),)
            #Timers.toc('setup')

            #Timers.tic('lpi.minimize')
            lp_result = self.lpi.minimize(lp_vec)
            if lp_result.dtype != dtype:
                lp_result = lp_result.astype(dtype)
            
            self.last_lp_result = lp_result
            
            self.num_lps += 1
            #Timers.toc('lpi.minimize')
            assert len(lp_result) == num_init_vars

            #print("--------")
            #print(f"lp_result: {lp_result}")

            #Timers.tic('a_mat mult')
            rv = np.dot(self.a_mat, lp_result) + self.bias
            #Timers.toc('a_mat mult')

        # return input as well
        if return_io:
            rv = [lp_result, rv]

        Timers.toc('star.minimize_vec')

        return rv

    def verts(self, xdim=0, ydim=1, epsilon=1e-7):
        'get a 2-d projection of this lp_star'

        dims = self.a_mat.shape[0]

        if isinstance(xdim, int):
            assert 0 <= xdim < dims, f"xdim {xdim} out of bounds for star with {dims} dims"
            vec = np.zeros(dims, dtype=float)
            vec[xdim] = 1
            xdim = vec
        else:
            assert xdim.size == dims

        if isinstance(ydim, int):
            assert 0 <= ydim < dims, f"ydim {ydim} out of bounds for star with {dims} dims"
            vec = np.zeros(dims, dtype=float)
            vec[ydim] = 1
            ydim = vec
        else:
            assert ydim.size == dims

        def supp_point_func(vec2d):
            'maximize a support function direction'

            Timers.tic('supp_point_func')

            # use negative to maximize
            lpdir = -vec2d[0] * xdim + -vec2d[1] * ydim

            res = self.minimize_vec(lpdir)

            Timers.toc('supp_point_func')

            # project onto x and y
            resx = np.dot(xdim, res)
            resy = np.dot(ydim, res)

            return np.array([resx, resy], dtype=float)

        Timers.tic('kamenev.get_verts')
        verts = kamenev.get_verts(2, supp_point_func, epsilon=epsilon)
        Timers.toc('kamenev.get_verts')

        #assert np.allclose(verts[0], verts[-1])
        
        return verts

    def box_verts(self, xdim=0, ydim=1):
        'get box bounds of a 2d projection of the star'

        dims = self.a_mat.shape[0]

        if isinstance(xdim, int):
            assert 0 <= xdim < dims, f"xdim {xdim} out of bounds for star with {dims} dims"
            vec = np.zeros(dims, dtype=float)
            vec[xdim] = 1
            xdim = vec
        else:
            assert xdim.size == dims

        if isinstance(ydim, int):
            assert 0 <= ydim < dims, f"ydim {ydim} out of bounds for star with {dims} dims"
            vec = np.zeros(dims, dtype=float)
            vec[ydim] = 1
            ydim = vec
        else:
            assert ydim.size == dims

        def supp_point_func(vec2d):
            'maximize a support function direction'

            Timers.tic('supp_point_func')

            # use negative to maximize
            lpdir = -vec2d[0] * xdim + -vec2d[1] * ydim

            res = self.minimize_vec(lpdir)

            Timers.toc('supp_point_func')

            # project onto x and y
            resx = np.dot(xdim, res)
            resy = np.dot(ydim, res)

            return np.array([resx, resy], dtype=float)

        # box bounds
        x0 = supp_point_func([1, 0])[0]
        x1 = supp_point_func([-1, 0])[0]
        y0 = supp_point_func([0, 1])[1]
        y1 = supp_point_func([0, -1])[1]

        verts = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
        
        return verts

    def execute_relus_overapprox(self, layer_num, layer_bounds, split_indices, zero_indices):
        '''
        run the relu part of the star update for the *entire* layer, using overapproximation
        '''

        Timers.tic('execute_relus_overapprox')

        # overapprox can set star to None if no LP optimization is needed
        num_outputs = self.a_mat.shape[0]
        assert len(layer_bounds) == num_outputs, f"outputs is {num_outputs}, but num " + \
            f"layer bounds {len(layer_bounds)}"

        #new_generators_bm = [[] for _ in range(num_outputs)]
        new_generators_bm = np.zeros((num_outputs, split_indices.size), dtype=self.a_mat.dtype)

        self.a_mat[zero_indices, :] = 0
        self.bias[zero_indices] = 0

        for i in split_indices:
            lb, ub = layer_bounds[i]
            self.split_overapprox(layer_num, new_generators_bm, i, lb, ub)

        Timers.tic('overapprox bm update')
        num_zeros = self.lpi.get_num_cols() - self.a_mat.shape[1]

        if num_zeros > 0:
            #for bm_row in new_generators_bm:
            #    bm_row += [0] * (num_zeros - len(bm_row))

            self.a_mat = np.hstack([self.a_mat, new_generators_bm])
            assert self.a_mat.shape[1] == self.lpi.get_num_cols()

        Timers.toc('overapprox bm update')

        Timers.toc('execute_relus_overapprox')

    def split_overapprox(self, layer_num, new_generators_bm, i, lb, ub):
        '''helper for execute_relus_overapprox

        split a ReLU using a star overapproximation'''

        Timers.tic('split_overapprox')

        # make a new variable y for the output
        self.lpi.add_positive_cols([f'y{layer_num}_{i}'])
        num_cols = self.lpi.get_num_cols()
        num_zeros = num_cols - self.a_mat.shape[1] - 1

        #zero_row = np.zeros((self.star.a_mat.shape[1],))

        # create 3 constraints for new variable
        # (1) y >= 0 === -y <= 0
        # this constraint is automatically added in lp_star for all non-cur variables

        # (2) y >= x[i] === x[i] - y <= 0
        # x[i] equals row i in the basis matrix (also the bias on the rhs)

        row = np.zeros((num_cols,), dtype=self.a_mat.dtype)
        a_mat_width = self.a_mat.shape[1]

        assert a_mat_width <= num_cols, f"a_mat_width: {a_mat_width}, num_cols: {num_cols}"

        row[:a_mat_width] = self.a_mat[i, :]
        row[-1] = -1
        self.lpi.add_dense_row(row, -self.bias[i])

        # (3) y <= ub*(x[i]-lb) / (ub - lb) === y - ub*x[i] / (ub - lb) - (ub*(-lb) / (ub - lb)) <= 0
        # === y - ub(ub - lb) * x[i] <= ub*(-lb) / (ub - lb)
        # x[i] equals row i in the basis matrix
        factor = ub / (ub - lb)
        row = np.zeros((num_cols,), dtype=self.a_mat.dtype)
        row[:self.a_mat.shape[1]] = -1 * factor * self.a_mat[i]
        row[-1] = 1
        rhs = -lb * factor + self.bias[i] * factor
        self.lpi.add_dense_row(row, rhs)

        # reset the current bias
        # the rhs of the current variable is not referenced by other constraints (constraints never ref rhs)
        self.bias[i] = 0

        # ReLU case, introduce new variable
        self.a_mat[i] = 0

        new_generators_bm[i, num_zeros] = 1

        Timers.toc('split_overapprox')

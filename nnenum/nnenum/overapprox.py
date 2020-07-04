'''
Overapproximation analysis
'''

import time
import numpy as np

from nnenum.settings import Settings
from nnenum.timerutil import Timers
from nnenum.util import Freezable
from nnenum.prefilter import update_bounds_lp, sort_splits
from nnenum.specification import DisjunctiveSpec
from nnenum.network import ReluLayer, FullyConnectedLayer, nn_flatten, nn_unflatten

def try_quick_overapprox(ss, network, spec):
    'try a quick overapproximation, return True if safe'
    
    overapprox_types = Settings.QUICK_OVERAPPROX_TYPES

    prerelu_sims = make_prerelu_sims(ss, network)

    rr = do_overapprox_rounds(ss, network, spec, prerelu_sims, overapprox_types=overapprox_types)

    return rr.is_safe

def make_prerelu_sims(ss, network):
    '''compute the prerelu simulation values at each remaining layer

    this only saves the state for the remaining layers, before relu is executed
    the output of the network is stored at index len(network.layers)

    returns a dict, layer_num -> sim_vector
    '''

    if ss.prefilter.simulation is None:
        rv = None
    else:
        rv = {}

        state = ss.prefilter.simulation[1].copy()
        layer_num = ss.cur_layer
        layer = network.layers[layer_num]

        rv[layer_num] = state

        # current layer may be partially processed
        if isinstance(layer, ReluLayer):
            state = np.clip(state, 0, np.inf)

        while layer_num + 1 < len(network.layers):
            layer_num += 1
            
            layer = network.layers[layer_num]
            rv[layer_num] = state

            shape = layer.get_input_shape()
            input_tensor = nn_unflatten(state, shape).astype(ss.star.a_mat.dtype)
            output_tensor = layer.execute(input_tensor)
            state = nn_flatten(output_tensor)

        # save final output
        rv[len(network.layers)] = state

    return rv

def check_round(ss, sets, spec_arg, check_cancel_func=None):
    '''check overapproximation result of one round against spec

    this may modify ss.safe_spec_list is part of the spec is proven as safe

    This returns is_safe?, violation_stars, violation_indices
    '''

    # todo: evaluate if trying to generate concrete counterexamples here is worth it
    Timers.tic('overapprox_check_round')

    if check_cancel_func is None:
        check_cancel_func = lambda: False
    
    whole_safe = True

    unsafe_violation_stars = [] # list of violation stars for each part of the disjunctive spec
    unsafe_violation_indices = [] # index in spec_list

    # break it apart disjunctive specs, as quicker overapproximation may work for some parts and not others
    spec_list = spec_arg.spec_list if isinstance(spec_arg, DisjunctiveSpec) else [spec_arg]

    for i, single_spec in enumerate(spec_list):

        if ss.safe_spec_list is not None and ss.safe_spec_list[i]:
            continue
        
        single_safe = False

        violation_star = None
        
        for s in sets:
            single_safe = s.check_spec(single_spec, check_cancel_func)

            if isinstance(s, StarOverapprox) and not single_safe:
                violation_star = s.violation_star

            #print(f".check_round checking spec with set {s}, result: {single_safe}")

            if single_safe:
                if ss.safe_spec_list is not None:
                    ss.safe_spec_list[i] = True
                
                break # done with this spec!

        if not single_safe:
            whole_safe = False

            if violation_star is not None:
                unsafe_violation_stars.append(violation_star)
                unsafe_violation_indices.append(i)

            if not Settings.CONTRACT_OVERAPPROX_VIOLATION:
                # if contracting violation, we need all violation stars
                break

    Timers.toc('overapprox_check_round')

    return whole_safe, unsafe_violation_stars, unsafe_violation_indices

class RoundsResult:
    'result of do_overapprox_rounds'

    def __init__(self):

        self.is_safe = False
        self.round_generators = [] # list of lists for each round
        self.round_ms = [] # ms for each round
        self.concrete_io_tuple = None

    def __str__(self):

        if Settings.SAVE_BRANCH_TUPLES_TIMES:
            rv = ", ".join([f"{max(r)} ({round(ms, 1)} ms)" for r, ms in zip(self.round_generators, self.round_ms)])
        else:
            rv = ", ".join([f"{max(r)}" for r in self.round_generators])
        
        # comma seperated for each round
        # {round(diff * 1000, 1)}
        # {round_max_gens} (100 ms)
        
        return rv

    def get_max_gens(self):
        'get the maximum number of generators from all the reprsentations'

        rv = -np.inf

        for gen_list in self.round_generators:
            rv = max(rv, max(gen_list))

        return rv
        
def do_overapprox_rounds(ss, network, spec, prerelu_sims, check_cancel_func=None, gen_limit=np.inf,
                         overapprox_types=None, try_seeded_adversarial=None):
    '''do the multi-round overapproximation analysis

    returns an instance of RoundsResult:
    'is_safe' -> bool
    'branch_label' -> list of 2-tuples for each round (max_generators [int], milliseconds [int])
    'max_gens' -> int

    returns (is_safe, branch_label. max_gens) 
    '''

    if overapprox_types is None:
        overapprox_types = Settings.OVERAPPROX_TYPES

    rv = RoundsResult()

    first_round = True
    sets = []

    if Settings.OVERAPPROX_CONTRACT_ZONO_LP:
        #if len(ss.branch_tuples) % 10 == 0:
        #    print(f"contracting lp randomly in overapprox")
        ss.prefilter.zono.contract_lp(ss.star)

    for round_num, types in enumerate(overapprox_types):
        assert isinstance(types, list), f"types was not list: {types}"
        sets.clear()

        for type_str in types:
            if type_str.startswith('zono.'):
                z = ZonoOverapprox(ss, type_str, gen_limit)
                sets.append(z)
            else:
                assert type_str.startswith('star.'), f"unknown type_str: {type_str}"
                s = StarOverapprox(ss, type_str, gen_limit)
                sets.append(s)

        start = time.perf_counter()

        try:
            run_overapprox_round(network, ss, sets, prerelu_sims, check_cancel_func)
            diff = time.perf_counter() - start if Settings.SAVE_BRANCH_TUPLES_TIMES else 0
        except OverapproxCanceledException as e:
            diff = time.perf_counter() - start if Settings.SAVE_BRANCH_TUPLES_TIMES else 0

            msg = f"canceled after {round(diff * 1000, 1)} ms"

            raise OverapproxCanceledException(f"{e}; {rv}, {msg}")

        #print(f".done with round {types} {round_num}/{len(overapprox_types)} in {round(diff * 1000, 1)} ms")

        gens = [s.get_num_gens() for s in sets]
        rv.round_generators.append(gens)
        rv.round_ms.append(diff * 1000)

        start = time.perf_counter()
        rv.is_safe, vstars, vindices = check_round(ss, sets, spec, check_cancel_func)

        #check_diff = time.perf_counter() - start
        #print(f"safe after round: {rv.is_safe}, runtime: {diff * 1000} ms, check: {check_diff * 1000} ms")

        if rv.is_safe:
             break

        if vstars and try_seeded_adversarial is not None and Settings.ADVERSARIAL_ONNX_PATH and \
                      Settings.ADVERSARIAL_FROM_ABSTRACT_VIO:
            Timers.tic('try_violation_adversarial')
            dims = len(np.ravel(Settings.ADVERSARIAL_ORIG_IMAGE))

            for vstar, vindex in zip(vstars, vindices):
                if isinstance(spec, DisjunctiveSpec):
                    row = spec.spec_list[vindex].mat[0, :]
                else:
                    row = spec.mat[0, :]

                cinput, coutput = vstar.minimize_vec(row, return_io=True)

                assert np.argmax(coutput) != Settings.ADVERSARIAL_ORIG_LABEL

                trimmed_inputs = cinput[:dims]
                shaped_input = nn_unflatten(trimmed_inputs, Settings.ADVERSARIAL_ORIG_IMAGE.shape)
                exec_output = network.execute(shaped_input)
                flat_output = np.ravel(exec_output)
                
                if np.argmax(flat_output) != Settings.ADVERSARIAL_ORIG_LABEL:
                    if Settings.PRINT_OUTPUT:
                        print("Found unsafe from concrete execution of abstract counterexample")
                        
                    rv.concrete_io_tuple = trimmed_inputs, flat_output
                else:
                    rv.concrete_io_tuple = try_seeded_adversarial(shaped_input)

                    if rv.concrete_io_tuple and Settings.PRINT_OUTPUT:
                        print("Found unsafe from adversarial search near abstract counterexample")

                if rv.concrete_io_tuple is not None:
                    break

            Timers.toc('try_violation_adversarial')

        if vstars and Settings.CONTRACT_OVERAPPROX_VIOLATION:
            ss.contract_from_violation(vstars)

        if first_round:
            first_round = False

    #print("done with overapprox rounds")

    #raise RuntimeError(".debug exit")
        
    return rv

def run_overapprox_round(network, ss_init, sets, prerelu_sims, check_cancel_func=None):
    '''
    run overapproximation analysis through the network (a single round with multiple sets)

    ss_init - the exact star set, at a split point in the network
    sets - a list of overapproximation set representations like ZonoOverapprox or StarOverapprox
    check_cancel_func - a function that can be called for long operations which may raise a OverapproxCanceledException

    this modifies the passed-in sets in place
    '''

    if check_cancel_func is None:
        check_cancel_func = lambda: False

    layer_num = ss_init.cur_layer
    depth = len(ss_init.branch_tuples)

    # precondition is that ss_init is at a split in the network
    assert layer_num < len(network.layers)
    assert isinstance(network.layers[layer_num], ReluLayer)
    assert ss_init.prefilter.output_bounds is not None
    assert ss_init.prefilter.output_bounds.branching_neurons.size > 0
    assert len(sets) > 0, "need at least one type of overapproximation set"

    split_indices = ss_init.prefilter.output_bounds.branching_neurons
    zero_indices = np.array([], dtype=int) # no zero assignments needed (already done eagerly)
    layer_bounds = ss_init.prefilter.output_bounds.layer_bounds

    #print(f". layer bounds {layer_num}:\n{layer_bounds}")

    # run first layer with existing bounds
    for s in sets:
        s.execute_with_bounds(layer_num, layer_bounds, split_indices, zero_indices)

    layer_num += 1

    # run remaining layers with newly-computed bounds
    remaining_layers = network.layers[layer_num:]
    
    for layer_index, layer in enumerate(remaining_layers):
        if Settings.NUM_PROCESSES == 1 and Settings.PRINT_OUTPUT:
            print(f"Running Overapprox layer {layer_index} / {len(remaining_layers)}: {type(layer)}")
            
        check_cancel_func()
        
        if isinstance(layer, ReluLayer):
            sim = None if prerelu_sims is None else prerelu_sims[layer_num]
            split_indices = None
            layer_bounds = None

            for s in sets:                    
                layer_bounds, split_indices = s.tighten_bounds(layer_bounds, split_indices, sim,
                                                               check_cancel_func, depth)

            #print(f". layer bounds {layer_num}:\n{layer_bounds}")

            # bounds are now as tight as they will get
            if split_indices is None:
                split_indices = make_split_indices(layer_bounds)
                
            split_indices = sort_splits(layer_bounds, split_indices)
            zero_indices = np.nonzero(layer_bounds[:, 1] < -Settings.SPLIT_TOLERANCE)[0]
            
            for s in sets:
                s.execute_with_bounds(layer_num, layer_bounds, split_indices, zero_indices)
        else:
            # non-relu layer
            Timers.tic('transform_linear')
            
            for s in sets:
                s.transform_linear(layer)

            Timers.toc('transform_linear')

        layer_num += 1

def make_split_indices(layer_bounds):
    'make split indices from layer bounds'

    Timers.tic('make_split_indices')
    split_indices = np.nonzero(np.logical_and(layer_bounds[:, 0] < -Settings.SPLIT_TOLERANCE, \
                                              layer_bounds[:, 1] > Settings.SPLIT_TOLERANCE))[0]
    Timers.toc('make_split_indices')

    return split_indices

class StarOverapprox(Freezable):
    '''
    star set (triangle) overapproximation set representation
    star sets support efficient affine transformation and intersection
    '''

    def __init__(self, ss, type_string, max_gens=np.inf):
        assert max_gens >= Settings.OVERAPPROX_MIN_GEN_LIMIT

        self.star = ss.star.copy()

        assert type_string in ['star.lp', 'star.quick']

        self.type_string = type_string
        self.do_lp = type_string == 'star.lp'

        self.violation_star = None # assigned in check_spec

        self.max_gens = max_gens

    def __str__(self):
        return f"[StarOverapprox ({self.type_string})]"

    def execute_with_bounds(self, layer_num, layer_bounds, split_indices, zero_indices):
        'do the layer overapproximation with the passed-in bounds'

        if self.get_num_gens() + len(split_indices) > self.max_gens:
            raise OverapproxCanceledException(f'star gens exceeds limit (> {self.max_gens})')

        self.star.execute_relus_overapprox(layer_num, layer_bounds, split_indices, zero_indices)

    def transform_linear(self, layer):
        'affine transformation'

        layer.transform_star(self.star)

    def tighten_bounds(self, layer_bounds, split_indices, sim, check_cancel_func, depth):
        '''
        update the passed-in layer bounds

        layer_bounds and/or split_indices may be None

        returns (layer_bounds, split_indices), split_indices can be None
        '''

        #print(f". split_indices on call (star): {split_indices}")
        #print(f". layer_bounds on call (star): {layer_bounds[:3]}")

        if layer_bounds is None:
            num_neurons = self.star.a_mat.shape[0]            
            layer_bounds = np.array([[-np.inf, np.inf] for _ in range(num_neurons)], dtype=float)
        elif split_indices is None:
            split_indices = make_split_indices(layer_bounds)

        #print(f". split_indices (Before lp): {split_indices}")

        if self.do_lp:
            both_bounds = Settings.OVERAPPROX_BOTH_BOUNDS

            split_indices = update_bounds_lp(layer_bounds, self.star, sim, split_indices, depth,
                                             check_cancel_func=check_cancel_func, both_bounds=both_bounds)

        return layer_bounds, split_indices

    def check_spec(self, spec, check_cancel_func):
        'returns issafe?'

        # todo: evaluate whether this helps
        check_cancel_func()

        self.violation_star = spec.get_violation_star(self.star)

        return self.violation_star is None

    def get_num_gens(self):
        'get the number of generators in the overapproximation (None if inapplicable)'

        return self.star.a_mat.shape[1]

class ZonoOverapprox(Freezable):
    '''
    Zonotope overapproximation
    '''

    def __init__(self, ss, type_string, max_gens=np.inf):
        '''
        initialize from an (exact) StarState,

        type_string is the type of overapproximation 'zono.area', 'zono.ybloat', or 'zono.interval'

        if gens exceeds max_gens, an OverapproxCanceledException is raised
        '''

        assert max_gens >= Settings.OVERAPPROX_MIN_GEN_LIMIT

        self.zono = ss.prefilter.zono.deep_copy()
        
        self.type_string = type_string
        self.max_gens = max_gens

        if type_string == 'zono.area':
            self.relu_update_func = relu_update_best_area_zono
        elif type_string == 'zono.ybloat':
            self.relu_update_func = relu_update_ybloat_zono
        else:
            assert type_string == 'zono.interval'
            self.relu_update_func = relu_update_interval_zono

    def __str__(self):
        return f"[ZonoOverapprox ({self.type_string})]"

    def execute_with_bounds(self, _layer_num, layer_bounds, split_indices, zero_indices):
        'do the layer overapproximation with the passed-in bounds'

        if self.get_num_gens() + len(split_indices) > self.max_gens:
            raise OverapproxCanceledException(f'{self.type_string} gens exceeds limit (> {self.max_gens})')

        update_zono(self.zono, self.relu_update_func, layer_bounds, split_indices, zero_indices)

    def transform_linear(self, layer):
        'affine transformation'

        layer.transform_zono(self.zono)

    def tighten_bounds(self, layer_bounds, _split_indices, _sim, _check_cancel_func, _depth):
        '''
        update the passed-in layer bounds

        layer_bounds and/or split_indices may be None

        returns (layer_bounds, split_indices), split_indices can be None
        '''

        box_bounds = self.zono.box_bounds()

        if layer_bounds is None:
            layer_bounds = box_bounds
        else:
            layer_bounds[:, 0] = np.maximum(layer_bounds[:, 0], box_bounds[:, 0])
            layer_bounds[:, 1] = np.minimum(layer_bounds[:, 1], box_bounds[:, 1])

        return layer_bounds, None

    def check_spec(self, spec, _check_cancel_func):
        'returns is_safe?'

        may_be_unsafe = spec.zono_might_violate_spec(self.zono)

        return not may_be_unsafe
        
    def get_num_gens(self):
        'get the number of generators in the overapproximation (-1 if inapplicable)'

        return self.zono.mat_t.shape[1]

def update_zono(z, relu_update_func, bounds, splits, zeros):
    'update a zono with the current bounds'

    # this assumes apply_linear_map was done first, so that only ReLU processing remains
    lb_len = bounds.shape[0]
    assert len(z.center) == lb_len, "zonotope dims ({len(z.center)}) doesn't match layer_bounds {lb_len}"

    gen_mat_t = z.mat_t
    center = z.center

    # these are the bounds on the input for each neuron in the current layer
    Timers.tic('assign_zeros')
    center[zeros] = 0
    gen_mat_t[zeros, :] = 0
    Timers.toc('assign_zeros')

    if splits.size > 0:
        new_generators = np.zeros((gen_mat_t.shape[0], len(splits)), dtype=z.dtype)

        Timers.tic('relu_update')
        for i, split_index in enumerate(splits):
            lb, ub = bounds[split_index]

            # need to add a new generator for the overapproximation
            relu_update_func(lb, ub, split_index, gen_mat_t, center, new_generators[:, i])
        Timers.toc('relu_update')

        Timers.tic('stack_new_generators')
        # need to update zonotope with new generators
        z.init_bounds += [(-1, 1) for _ in range(len(splits))]

        z.mat_t = np.hstack([z.mat_t, new_generators])

        Timers.toc('stack_new_generators')

def relu_update_interval_zono(_lb, ub, output_dim, gen_mat_t, center, new_gen):
    '''update one dimension (output) of a zonotope due to a relu split
    This function produces the interval zonotope.
    '''

    gen_mat_t[output_dim] = 0
    
    y_offset = ub / 2.0
    center[output_dim] = y_offset

    new_gen[output_dim] = y_offset

def relu_update_ybloat_zono(lb, _ub, output_dim, _gen_mat_t, center, new_gen):
    '''update one dimension (output) of a zonotope due to a relu split
    This function produces the ybloat zonotope (new generator is vertical).
    '''

    y_offset = -lb / 2.0

    center[output_dim] += y_offset
    new_gen[output_dim] = y_offset

def relu_update_best_area_zono(lb, ub, output_dim, gen_mat_t, center, new_gen):
    '''update one dimension (output) of a zonotope due to a relu split
    This function produces the best-area zonotope.
    '''

    assert lb < Settings.SPLIT_TOLERANCE
    assert ub > -Settings.SPLIT_TOLERANCE

    slope_lambda = ub / (ub - lb) 
    gen_mat_t[output_dim] *= slope_lambda

    # add new generator value to bm
    mu = -1 * (ub * lb) / (2 * (ub - lb))
    new_gen[output_dim] = mu

    # modify center
    center[output_dim] = center[output_dim] * slope_lambda + mu

class OverapproxCanceledException(Exception):
    'an exception used for when overapproximation analysis is canceled'

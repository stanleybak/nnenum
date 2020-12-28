'''
adversarial generation for nnenum

Stanley Bak, 2020
'''

import os
import logging
import time
import warnings

import numpy as np

import onnx

import tensorflow as tf
import foolbox as fb

from nnenum.util import Freezable
from nnenum.timerutil import Timers
from nnenum.network import nn_unflatten
from nnenum.settings import Settings

class AgenState(Freezable):
    'adversarial image generation container'

    def __init__(self, onnx_filename, orig_image, label, epsilon, bounds=(0.0, 1.0)):
        '''initialize a session'''

        warnings.filterwarnings("ignore", category=UserWarning)
        logging.basicConfig(level=logging.ERROR)
        #warnings.filterwarnings("ignore", message='exponential search failed')

        # turn of logging errors
        #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        logging.getLogger('tensorflow').setLevel(logging.FATAL)

        # disable eager execution
        tf.compat.v1.disable_eager_execution()

        # slightly hack... onnx model is used only to get input / output names
        assert onnx_filename.endswith(".onnx")
        model = onnx.load(onnx_filename)
        self.output_name = f"{model.graph.output[0].name}:0"
        self.input_name = f"{model.graph.input[0].name}:0"

        filename = f'{onnx_filename}.pb'
        graph_def = None

        with tf.io.gfile.GFile(filename, 'rb') as f: 
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        assert graph_def is not None

        self.graph_def = graph_def
        self.orig_image = orig_image
        self.epsilon = epsilon
        self.bounds = bounds
        self.labels = np.array([label])

        self.sess = tf.compat.v1.Session()

        with self.sess.as_default():
            tf.import_graph_def(self.graph_def, name='')
            graph = self.sess.graph

            input_tensor = graph.get_tensor_by_name(self.input_name)
            output_tensor = graph.get_tensor_by_name(self.output_name)

            self.fmodel = fb.models.TensorFlowModel(input_tensor, output_tensor, bounds=self.bounds)

        self.freeze_attrs()

    def __del__(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def try_mixed_adversarial(self, iteration, random_only):
        '''
        try generating an adversarial using a mixed strategy, depending on the iteration

        returns [adversarial image, epsilon], if found, else None
        '''

        rv = None

        classes = [fb.attacks.FGSM, # 0.057 in 41ms
         fb.attacks.ContrastReductionAttack, # 0.05 in 64 ms
         fb.attacks.BlendedUniformNoiseAttack, # 0.09 in 93ms
         fb.attacks.DecoupledDirectionNormL2Attack, # 0.074 in 124ms
         fb.attacks.BIM, # 0.044 in 300 ms
         fb.attacks.PGD, # 0.05 in 1302 ms
         fb.attacks.MomentumIterativeAttack, # 0.04 in 300 ms
         fb.attacks.AdamPGD, #0.055 in 800ms
         fb.attacks.AdamRandomPGD, # 0.042 in 700ms
         fb.attacks.RandomPGD # best
         ]

        # pick the attack class...
        attack_class = None
        
        if not random_only and iteration < len(classes):
            attack_class = classes[iteration]

        t = Settings.ADVERSARIAL_TARGET
        criterion = fb.criteria.Misclassification() if t is None else fb.criteria.TargetClass(t)
            
        with self.sess.as_default():
            if attack_class is None:
                attack_class = SingleEpsilonRPGD
                attack = SingleEpsilonRPGD(self.fmodel, distance=fb.distances.Linfinity, criterion=criterion)

                # subtract a small amount since attack was overshooting by numerical precision
                SingleEpsilonRPGD.set_epsilon(self.epsilon - 1e-6)
            else:
                attack = attack_class(self.fmodel, distance=fb.distances.Linfinity, criterion=criterion)

            Timers.tic('attack')
            a = attack(self.orig_image, self.labels, unpack=False)[0]
            Timers.toc('attack')

            dist = a.distance.value

            #print(f"attack class: {attack_class}, ep: {dist}, iteration {iteration}")

            if dist <= Settings.ADVERSARIAL_EPSILON:
                rv = a.perturbed
                rv.shape = self.orig_image.shape

        if rv is not None:
            print(f"try_mixed_adversarial found violation image on iteration {iteration} with ep={dist} and " + \
                  f"attack class: {attack_class}")

        return rv

    def try_single(self):
        '''try to generate an adversarial image for the single value of epsilon (quick)

        returns [adversarial image, epsilon], if found, else None
        '''

        Timers.tic('try_single')

        rv = None

        t = Settings.ADVERSARIAL_TARGET
        criterion = fb.criteria.Misclassification() if t is None else fb.criteria.TargetClass(t)

        with self.sess.as_default():
            attack = SingleEpsilonRPGD(self.fmodel, distance=fb.distances.Linfinity, criterion=criterion)

            # subtract a small amount since attack was overshooting by numerical precision
            SingleEpsilonRPGD.set_epsilon(self.epsilon - 1e-6)

            Timers.tic('attack')
            a = attack(self.orig_image, self.labels, unpack=False)[0]
            Timers.toc('attack')

            dist = a.distance.value

            if dist != np.inf:
                rv = [a.perturbed, dist]
                rv[0].shape = self.orig_image.shape

        Timers.toc('try_single')

        return rv

    def try_seeded(self, seed_image):
        '''try to generate the closest adversarial image from a given seed image, and check if it's within
        the desired epsilon from the original image

        returns [adversarial image, epsilon], if found, else None
        '''

        Timers.tic('try_seeded')

        assert seed_image.shape == self.orig_image.shape

        rv = None

        closest_status = ""
        closest_dist = np.inf

        classes = [
         #fb.attacks.FGSM, # 0.057 in 41ms
         #fb.attacks.ContrastReductionAttack, # 0.05 in 64 ms
         #fb.attacks.BlendedUniformNoiseAttack, # 0.09 in 93ms
         #fb.attacks.NewtonFoolAttack, # 0.11 in 93ms
         #fb.attacks.DecoupledDirectionNormL2Attack, # 0.074 in 124ms
         fb.attacks.BIM, # 0.044 in 300 ms
         #fb.attacks.PGD, # 0.05 in 1302 ms
         #fb.attacks.MomentumIterativeAttack, # 0.04 in 300 ms
         #fb.attacks.AdamPGD, #0.055 in 800ms
         #fb.attacks.AdamRandomPGD, # 0.042 in 700ms
         #fb.attacks.RandomPGD # best
         ]

        t = Settings.ADVERSARIAL_TARGET
        criterion = fb.criteria.Misclassification() if t is None else fb.criteria.TargetClass(t)
        
        with self.sess.as_default():
            for attack_class in classes:

                #if rv is not None:
                #    break
                
                attack = attack_class(self.fmodel, distance=fb.distances.Linfinity, criterion=criterion)

                factor = 0.65 # sweet spot

                blended_image = factor * seed_image + (1.0 - factor) * self.orig_image

                # clip to fix floating-point out of bounds
                blended_image = np.clip(blended_image, self.bounds[0], self.bounds[1])

                #blended_ep = np.linalg.norm(np.ravel(self.orig_image - blended_image), ord=np.inf)

                #if blended_ep > Settings.ADVERSARIAL_EPSILON:
                #    break

                start = time.perf_counter()
                Timers.tic('attack')
                a = attack(blended_image, self.labels, unpack=False)[0]
                Timers.toc('attack')

                diff = time.perf_counter() - start

                if a.distance.value != np.inf:
                    # compute the distance from the original imag
                    aimage = a.perturbed
                    aimage.shape = self.orig_image.shape

                    diff_image = self.orig_image - aimage

                    dist = np.linalg.norm(np.ravel(diff_image), ord=np.inf)

                    status = f"({factor}) {round(diff * 1000, 1)}ms, {round(a.distance.value, 5)} from " + \
                             f"blended and {round(dist, 5)} from orig with attack class: {attack_class.__name__}"

                    #print(f".agen {status}")

                    if dist < closest_dist:
                        closest_dist = dist
                        closest_status = status

                    tol = 1e-6

                    if dist <= self.epsilon + tol:
                        rv = [aimage, dist]

                        print(f"SUCCESS: seed succeeded with factor {factor}.")
                        break

        label = "SEED-SUCCESS" if rv is not None else "SEED-FAILED"
        print(f"{label}: {closest_status}", flush=True)
                        
        Timers.toc('try_seeded')

        return rv

class SingleEpsilonRPGD(
    fb.attacks.iterative_projected_gradient.LinfinityGradientMixin,
    fb.attacks.iterative_projected_gradient.LinfinityClippingMixin,
    fb.attacks.iterative_projected_gradient.LinfinityDistanceCheckMixin,
    fb.attacks.iterative_projected_gradient.GDOptimizerMixin,
    fb.attacks.iterative_projected_gradient.IterativeProjectedGradientBaseAttack,
):
    'random projected gradient descent with custom parameters'

    epsilon = 0 # bad... storing in class variable

    @classmethod
    def set_epsilon(cls, epsilon):
        'set epsilon value'
        cls.epsilon = epsilon

    @fb.attacks.base.generator_decorator
    def as_generator(
        self,
        a,
        binary_search=False,
        epsilon=None, # use from constructor
        stepsize=0.01,
        iterations=50,
        random_start=True,
        return_early=True,
    ):
        epsilon = SingleEpsilonRPGD.epsilon
        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )

def try_quick_adversarial(num_attempts, remaining_secs=None):
    '''try a quick adversarial example using Settings

    returns AgenState instance, aimage (may be None)
    '''

    start = time.perf_counter()

    onnx_path = Settings.ADVERSARIAL_ONNX_PATH
    ep = Settings.ADVERSARIAL_EPSILON
    im = Settings.ADVERSARIAL_ORIG_IMAGE
    l = Settings.ADVERSARIAL_ORIG_LABEL

    agen = AgenState(onnx_path, im, l, ep)
    a = None

    for i in range(num_attempts):

        if remaining_secs is not None:
            diff = time.perf_counter() - start
            
            if diff > remaining_secs:
                break # timeout!
        
        a = agen.try_single()

        if a is not None:
            break

    if a is not None:
        aimage, ep = a

        if Settings.PRINT_OUTPUT and num_attempts > 0:
            print(f"try_quick_adversarial found violation image on iteration {i} with ep={ep}")
        
    else:
        aimage = None

    #with agen.sess.as_default():
    #    attack = fb.attacks.RandomPGD(agen.fmodel, distance=fb.distances.Linfinity)

    #    Timers.tic('attack')
    #    a = attack(agen.orig_image, agen.labels, unpack=False)[0]
    #    Timers.toc('attack')

    #    dist = a.distance.value

    #    print(f"\nDist of random PGD adversarial: {dist}")
        
    return agen, aimage

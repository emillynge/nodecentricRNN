__author__ = 'emil'
import numpy as np
import logging
from functools import wraps, partial
from collections import namedtuple, defaultdict
from . import nodes

class SigmoidDerivativeCache(object):
    derivatives = dict()

    def __call__(self, _input):
        if id(_input) not in self.derivatives:
            self.derivatives[id(_input)] = self.sigmoid_derivative(_input)
        return self.derivatives[id(_input)]

    def clear(self):
        self.derivatives.clear()

    @staticmethod
    def sigmoid_derivative(sig_of_x):
        """ calculate derivative of sigmoid(x), given the result of sigmoid(x) """
        sig_of_x = np.matrix(sig_of_x)
        return sig_of_x - np.power(sig_of_x, 2)

sigmoid_derivative = SigmoidDerivativeCache.sigmoid_derivative


class Theta(object):
    def __init__(self, num_in, num_out, *hidden_layer_sizes, theta_vec=None, mu=.5, sigma=1, zeros=False):
        self.construct_args = tuple([num_in, num_out] + list(hidden_layer_sizes))
        self.layer_sizes = [num_in] + list(hidden_layer_sizes) + [num_out]

        """ define connection shapes
             direct connections:    L_{i-1, t} -> L_{i, t}      forall i in 1..N-1
             context connections:   L_{i, t - 1} -> L_{i, t}    forall i in 1..N-1
             output_connections:    L_{i,t} -> L_{N, t}         forall i in 1..N-1
             bias_connections:     [1] -> L_{i, t}             forall i in 1..N

             shape is such that connections W are applied as:
                L2 = W_(L1->L2) @ L1
                i.e L2_W_L1
                L1.shape = (n_1, 1)
        """
        self.direct_conn_shapes = [(n, p) for p, n in zip(self.layer_sizes[:-2], self.layer_sizes[1:-1])]
        self.context_conn_shapes = [(hidden_size, hidden_size) for hidden_size in self.layer_sizes[1:-1]]
        self.output_conn_shapes = [(num_out, hidden_size) for hidden_size in self.layer_sizes[1:-1]]
        self.in2out_conn_shapes = [(num_out, num_in)]
        self.bias_conn_shapes = [(hidden_size, 1) for hidden_size in self.layer_sizes[1:]]

        self.n_weights = sum(np.prod(shape) for shape in self.direct_conn_shapes + self.context_conn_shapes
                             + self.output_conn_shapes + self.bias_conn_shapes + self.in2out_conn_shapes)
        self.n_hidden = len(hidden_layer_sizes)


        if theta_vec:
            if len(theta_vec) != self.n_weights:
                raise ValueError(
                    "Given layer sizes {0} theta_vec should have length {2}, not {3}".format(self.layer_sizes,
                                                                                             self.n_weights,
                                                                                             theta_vec))
            self.constructor = partial(np.array, theta_vec, dtype=float)
        elif zeros:
            self.constructor = partial(np.zeros, (self.n_weights,))
        else:
            # initialize connection weights in a vector
            # noinspection PyArgumentList

            self.constructor = partial(np.random.normal, loc=mu, scale=sigma, size=self.n_weights)

        self.vector = self._make_vector()
        # provide views to all connections
        self.direct_weights, i = self.construct_weight_views(self.direct_conn_shapes)
        self.context_weights, i = self.construct_weight_views(self.context_conn_shapes, i)
        self.output_weights, i = self.construct_weight_views(self.output_conn_shapes, i)
        self.bias_weights, i = self.construct_weight_views(self.bias_conn_shapes, i)
        self.in2out_weights, i = self.construct_weight_views(self.in2out_conn_shapes, i)

    def update_vector(self, new_vector):
        if len(new_vector) != len(self.vector):
            raise ValueError('vector length mismatch')
        #for (i, new_weight) in enumerate(new_vector):
        #    self.vector[i] = new_weight
        self.vector[:] = new_vector[:]

    def _make_vector(self):
        return np.array(self.constructor()).reshape((self.n_weights,))

    def reset_vector(self):
        self.vector[:] = self._make_vector()[:]

    def deepcopy(self):
        vector = list(self.vector.tolist())
        return Theta(*self.construct_args, theta_vec=vector)

    def zeros_copy(self):
        return self.__class__(*self.construct_args, zeros=True)

    def __setitem__(self, index, value):
        self.vector[index] = value

    def __delitem__(self, index):
        del self.vector[index]

    def __iter__(self):
        return self.vector.__iter__()

    def insert(self, index, value):
        raise NotImplementedError("You don't want to do this!")

    def construct_weight_views(self, shapes, i=0):
        weight_list = list()
        for shape in shapes:
            n = np.prod(shape)
            weight_list.append(self.vector[i:(i + n)].reshape(shape))
            i += n
        return weight_list, i

    def __len__(self):
        return len(self.layer_sizes) - 1

    def __getitem__(self, item):
        return self.vector[item]

    def weights(self, layer):
        if layer == len(self.direct_weights) or layer == -1:  # We are at output layer!
            return np.concatenate(self.output_weights + [self.bias_weights[-1]], axis=1)
        return np.concatenate((self.direct_weights[layer], self.context_weights[layer], self.bias_weights[layer]),
                              axis=1)


class ThetaNodeGenerator(object):
    def __init__(self, *args, **kwargs):
        self.weights = Theta(*args, **kwargs)
        self.gradients = Theta(*args, zeros=0)
        self.nodes = defaultdict(dict)

    def __call__(self, weight_type, layer):
        try:
            return self.nodes[weight_type][layer]
        except KeyError:
            self.nodes[weight_type][layer] = self.make_new_node(weight_type, layer)
            return self.nodes[weight_type][layer]

    def make_new_node(self, weight_type, layer):
        weight = getattr(self.weights, weight_type + '_weights')[layer]
        gradient = getattr(self.gradients, weight_type + '_weights')[layer]
        return nodes.ParameterNode(weight, gradient, name=NodeName('theta_' + weight_type, l=layer))


def debug(self, msg, level='debug'):
    _msg = '{0!r:40} - '.format(self) + msg
    if level == 'debug':
        logging.debug(_msg)
    elif level == 'warning':
        logging.warning(_msg)
    elif level == 'info':
        logging.info(_msg)


def conform(self_protocol, *input_protocols):
    """
    Checks the basic class assumptions required by a mixin
    :param self_protocol: The protocol that the inheriting class should adhere to
    :param input_protocols: An iterable of protocols that inputs should adhere to

    This is mostly practical for coding in an IDE that uses isinstance asserts to do autocompletion
    """
    def conformity_wrapper(bound_method):
        @wraps(bound_method)
        def check_for_class_conformity_and_execute(self, *inputs, **kwargs):

            assert isinstance(self, self_protocol)     # self must be a parent node
            for (inp, protocol) in zip(inputs, input_protocols):
                assert isinstance(inp, protocol)
            return bound_method(self, *inputs, **kwargs)
        return check_for_class_conformity_and_execute
    return conformity_wrapper

NodeName = namedtuple('NodeName', 'var t l')
NodeName.__new__.__defaults__ = ('node', None, None)


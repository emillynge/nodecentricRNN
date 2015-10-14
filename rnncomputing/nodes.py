__author__ = 'emil'
from abc import (ABC, abstractmethod)
from functools import (wraps, lru_cache)
from operator import itemgetter
from collections import defaultdict, Sequence
import numpy as np
from .utils import (sigmoid_derivative, debug)
import inspect
from functools import partial
import logging
import warnings
eps = np.finfo(float).eps * 10

class BackPropNotReady(LookupError):
    pass

class GradientCheckError(ValueError):
    pass

class StandardOperatorHandler(object):
    def __add__(self, other):
        return AdditionNode(self, other)

    def __radd__(self, other):
        return AdditionNode(other, self)

    def __matmul__(self, other):
        return DotNode(self, other)

    def __rmatmul__(self, other):
        return DotNode(other, self)


class Node(ABC):
    operator_handler = StandardOperatorHandler

    def __getattr__(self, item):
        if item[:2] == '__' and hasattr(self.operator_handler, item):
            return getattr(self, item)

    def __repr__(self):
        return type(self).__name__ + '(' + self.name + ')'

    def __init__(self, *input_nodes, name: tuple=()):
        self.do_backprop = False
        self.touched = False
        for node in input_nodes:
            assert isinstance(node, Node)
        self.input_nodes = list(input_nodes)
        self.parent_nodes = list()
        self.parent_gradients = dict()
        self._total_cost_gradient = None
        self._output = None
        self._name = name

        function_names = set(list(zip(*inspect.getmembers(self.__class__, inspect.isfunction)))[0])
        for name, method in inspect.getmembers(self.operator_handler, inspect.isfunction):
            if name not in function_names:
                setattr(self.__class__, name, method)
        self.__post_init_touched = False
        self.post_init()
        if not self.__post_init_touched:
            raise NotImplementedError("""Node.post_init has not been touched.
            Remember to call super in post_init overrides!""")

    @property
    def name(self) -> str:
        if self._name:
            subscripts = [str(subscr) for subscr in self._name[1:] if subscr is not None]
            subscript = '[' + ','.join(subscripts) + ']' if subscripts else ''
            return self._name[0] + subscript
        return ', '.join(inp.name for inp in self.input_nodes)

    def post_init(self) -> None:
        """
        End point for post_inits.
        :return:
        """
        self.__post_init_touched = True

    @property
    def input_nodes_iter(self):
        for input_node in self.input_nodes:
            assert isinstance(input_node, Node)
            yield input_node


    @abstractmethod
    def compute_cost_gradient_wrt_input_node(self, input_node_idx: int) -> np.matrix:
        """
        This method can assume that all input nodes has a output property. And that total cost gradient is available
        :param input_node_idx:
        :return:
        """
        d_cost_d_input_node = np.matrix([[0]])
        return d_cost_d_input_node

    @abstractmethod
    def compute_output(self, *inputs) -> np.matrix:
        """
        :param inputs: in iterable of np.matrix *NOT* input nodes
        :return: output matrix
        """
        output = np.matrix([[0]])
        return output

    @property
    def total_cost_gradient(self) -> np.matrix:
        if self._total_cost_gradient is None:
            try:
                self._total_cost_gradient = sum(self.parent_gradients[parent_node] for parent_node in self.parent_nodes)
            except LookupError:
                raise BackPropNotReady('Some parents have not reported their gradient contribution')
        return self._total_cost_gradient

    @property
    def output(self) -> np.matrix:
        """
        Simple mirror method that can be overridden to change output behaviour
        :return: np.matrix
        """
        if self._output is None:
            raise BackPropNotReady('{!r} Node output not calculated!'.format(self))
        return self._output

    def post_forward_prop(self):
        """
        Cleanup after forward propagation
        Mostly in preparation for the back propagation that will follow
            Clear gradients so backprop know to recalculate
        :return: None
        """
        self._total_cost_gradient = None
        self.parent_gradients.clear()

    def post_backprop(self):
        """
        Cleanup after successful backpropagation
        Mostly in preparation for the forward propagation that will follow
            - Reset output so forward prop knows to recalculate
            - Set touched to False so forwardprop can accurately tag the nodes that have been touched
        :return:
        """
        self._output = None
        self.touched = False

    def recur_reset(self):
        self.post_backprop()
        self.post_forward_prop()

        for inp in self.input_nodes_iter:
            inp.recur_reset()

    def backprop(self, cost_gradient_contribution: np.matrix, parent_node):
        """
        Back propagate a gradient contribution
            Almost exclusively called by parent node.
            Will *not* back propagate until all gradient contributions from *touched* parents have been collected
        :param cost_gradient_contribution: dC/dParent * dParent/dSelf
        :param parent_node: Node for which self provides input that influences cost function
        :return: None
        """
        debug(self, 'Backpropagating')
        self.parent_gradients[parent_node] = cost_gradient_contribution
        try:
            for i, input_node in enumerate(self.input_nodes_iter):
                if input_node.do_backprop:
                    debug(self, 'propagating to {!r}'.format(type(input_node).__name__))
                    d_cost_d_input_node = self.compute_cost_gradient_wrt_input_node(i)
                    input_node.backprop(d_cost_d_input_node, self)

            #   Code only reached if ALL parent nodes have calculated the cost gradient wrt. to this node
            self.post_backprop()

        except BackPropNotReady as e:
            debug(self, 'Waiting for a parent', level='debug')
            for parent in self.parent_nodes:
                if parent not in self.parent_gradients:
                    debug(parent, 'Not ready', level='debug')

    def forward_prop(self) -> np.matrix:
        """
        Start forward propagation to compute node outputs
            Is called from top to bottom, *but* is calculated from bottom to top, hence *forward* propagation
            Input from all child nodes is collected before output is computed and forward propagated to parent that made
            the call.
        :return: self.output -> np.matrix
                    Note that this property method can alter the contents of ._output that is calculated in forward prop
        """
        self.touched = True         # Mark node for back propagation
        if self._output is None:
            inputs = [input_node.forward_prop() for input_node in self.input_nodes_iter]
            self._output = self.compute_output(*inputs)
        self.post_forward_prop()
        return self.output

    def recur_check_grad(self, cost_node, check_set: set, h=0.001, tol=.001):
        if self.do_backprop and self not in check_set:
            check_set.add(self)
            diff = self.central_approx_node(cost_node, h=h)
            debug(self, 'diff = {:1.2e}'.format(diff), level='info')
            if diff > tol:
                self.central_approx_node(cost_node, h=h)
                raise GradientCheckError('{0!r}: gradient difference of {1} above limit of {2}'.format(self, diff, tol))
            for inp in self.input_nodes_iter:
                inp.recur_check_grad(cost_node, check_set, h=h, tol=tol)

    def central_approx_node(self, cost_node, h=0.001) -> float:
        """
        Calculate the central approximation difference of cost wrt. this node
        :param cost_node: TopNode with scalar output
        :param h: step scalar
        :return: infinity-norm of the difference between total_gradient and approximated gradient
        """
        # Initialize network
        cost_node.recur_reset()
        cost_node.forward_prop()

        # collect original outputs and gradients
        orig_output = np.matrix(self._output.tolist())
        cost_node.start_backprop()
        orig_gradient = self._total_cost_gradient
        approx_gradient = np.zeros(orig_gradient.shape)

        def cost(i,j,val):
            """ output of cost_node if out[i,j] is set to val
            """
            orig_output[i,j] = val
            self._output = orig_output
            cost_node.forward_prop()
            J = cost_node.output[0, 0]
            cost_node.recur_reset()
            return J

        # Change elements of output matrix one at a time to get central difference wrt. that element
        for i, row in enumerate(orig_output.tolist()):
            for j, ele in enumerate(row):
                J_f = cost(i,j, ele + h)
                J_b = cost(i,j, ele - h)
                approx_gradient[i,j] = (J_f - J_b) / (2 * h)
                orig_output[i,j] = ele

        diff_mat = approx_gradient - orig_gradient
        max_diff = np.max(np.min(np.array([np.divide(diff_mat, orig_gradient).flatten().tolist(),
                                    np.divide(diff_mat, approx_gradient).flatten().tolist()]), axis=0))
        return abs(max_diff)


class InputNode(Node):
    """
    Concrete implementation of the Node protocol that denotes a leaf Node with *no* inputs
        output is *always* input_matrix and it should *never* be given input nodes
    """
    def compute_output(self, *inputs):
        """
        Return input matrix
        :param inputs: Should always be empty iterable
        :return: input matrix
        """
        return np.matrix(self.__input_matrix)

    def compute_cost_gradient_wrt_input_node(self, input_node_idx):
        """
        Dummy method as this is a leaf node with no input
            If you want to override this, you should really subclass from Node itself!
        :param input_node_idx:
        :return: None
        """
        raise ValueError("An {!r} should have no input nodes".format(type(self).__name__))

    def __init__(self, input_matrix: np.ndarray, name: tuple=tuple()):
        self.__input_matrix = input_matrix
        super(InputNode, self).__init__(name=name)

    @property
    def output(self) -> np.matrix:
        if self._output is not None:
            return self._output
        return np.matrix(self.__input_matrix)


class ParameterNode(InputNode):
    """
    Subclass of InputNode that implements bookkeeping of cost gradient and sets do_backprop to True
        Default is that non-parameter nodes do not back propagate unless they are ancestors to a ParameterNode
    """
    def __init__(self, parameter_matrix, gradient_matrix, name: tuple=tuple()):
        self.gradient_matrix = gradient_matrix
        super(ParameterNode, self).__init__(parameter_matrix, name=name)

    def post_init(self):
        self.do_backprop = True
        super(ParameterNode, self).post_init()

    def backprop(self, *args, **kwargs):
        super(ParameterNode, self).backprop(*args, **kwargs)
        try:
            gradient = super(ParameterNode, self).total_cost_gradient
            self.gradient_matrix[:, :] = gradient
        except BackPropNotReady:
            pass


class GroundTruthNode(InputNode):
    """
    Carbon copy of InputNode. The naming of the class is a matter of readability
    """
    pass


class ParentNode(Node):
    """
    Absstract subclass of Node, Implements functionality for nodes that has children
    """
    def post_init(self):
        self.do_back_backprop_if_children_does()
        self.register_as_parent()
        super(ParentNode, self).post_init()

    def do_back_backprop_if_children_does(self):
        self.do_backprop = False
        for input_node in self.input_nodes_iter:
            if input_node.do_backprop:
                self.do_backprop = True
                break


    def register_as_parent(self):
        for input_node in self.input_nodes_iter:
            input_node.parent_nodes.append(self)


class MutableInputMixin(object):
    """
    Mixin class that can be applied to Nodes that can handle arbitrary number of input nodes
        Implementation assumes that Concrete implementation of Node is subclass of ParentNode!
    """
    def check_for_class_conformity(self, input_node):
        """
        Checks the basic class assumptions required by a mixin
            self must be an implementation of Parentnode
            input_node must be an implementation of Node
        This is mostly practical for coding in an IDE that uses isinstance asserts to do autocompletion
        """
        assert isinstance(self, ParentNode)     # self must be a parent node
        assert isinstance(input_node, Node)     # input_node must be a concrete Node
        return self, input_node

    def append(self: ParentNode, input_node: Node):
        """
        Add new node to inputs
        :return:
        """
        self, input_node = self.check_for_class_conformity(input_node)

        if self in input_node.parent_nodes:
            raise NotImplementedError("Node is already a parent of this input node. Currently not supported")
        "@self: ParentNode"
        self.input_nodes.append(input_node)
        input_node.parent_nodes.append(self)

        if input_node.do_backprop:
            self.do_backprop = True     # if child needs backprop, so does this node

    def remove(self: ParentNode, input_node: Node):
        self, input_node = self.check_for_class_conformity(input_node)

        self.input_nodes.remove(input_node)
        input_node.parent_nodes.remove(self)

        if input_node.do_backprop and self.do_backprop:
            # if removed child needed backprop we might not need it any longer for this node
            self.do_back_backprop_if_children_does()

    def extend(self, input_nodes: Sequence):
        for node in input_nodes:
            self.append(node)

    def __iadd__(self, other):
        if not isinstance(other, Sequence):
            other = [other]
        self.extend(other)
        return self

    def __add__(self, other):
        assert isinstance(self, ParentNode)
        if not isinstance(other, Sequence):
            other = [other]

        new_node = self.__class__(*self.input_nodes)
        new_node += other
        return new_node


class AdditionNode(ParentNode, MutableInputMixin):
    def compute_output(self, *inputs):
        return sum(inputs)

    def compute_cost_gradient_wrt_input_node(self, input_node_idx: int) -> np.matrix:
        return self.total_cost_gradient


class DotNode(ParentNode):
    def compute_output(self, left_matrix, right_matrix):
        assert isinstance(left_matrix, np.matrix)
        assert isinstance(right_matrix, np.matrix)
        return left_matrix @ right_matrix

    def compute_cost_gradient_wrt_input_node(self, input_node_idx):
        """

        O = L @ R
                        R11 R12
                        R21 R22
                        R31 R32

        L11 L12 L13     O11 O12
        L21 L22 L23     O21 O22

        dC/dO = dC/dO11 dC/dO12
                dC/dO21 dC/dO22

        dO/dL = [R11 R21 R31    [R12 R22 R32
                 0   0   0]      0   0   0]

                 [0  0   0      [0  0   0
                 R11 R21 R31]   R12 R22 R32]

        dC/dL = dC/dO11 * [R11 R21 R31  + dC/dO12 * [R12 R22 R32 + \
                          0   0   0]                 0   0   0]
                dC/dO21 * [0  0   0     + dC/dO22 * [0  0   0
                          R11 R21 R31]               R12 R22 R32]

              = [R11 * dC/dO11 + R12 * dC/dO12   R21 * dC/dO11 + R22 * dC/dO12   R31 * dC/dO11 + R32 * dC/dO12
                 R11 * dC/dO21 + R12 * dC/dO22   R21 * dC/dO11 + R22 * dC/dO22   R31 * dC/dO11 + R32 * dC/dO22]

              = dC/dO @ R.T

        dO/dR = [L11 0   [0 L11
                 L12 0    0 L12
                 L13 0]   0 L12]

                 [L21 0  [0 L21
                  L22 0   0 L22
                  L23 0]  0 L23

        dC/dR = [L11 * dC/dO11 + L21 * dC/dO21   L11 * dC/dO12 + L21 * dC/dO22
                 L12 * dC/dO11 + L22 * dC/dO21   L12 * dC/dO12 + L22 * dC/dO22
                 L13 * dC/dO11 + L23 * dC/dO21   L13 * dC/dO12 + L23 * dC/dO22

              = L.T @ dC/dO

        :param input_node_idx:
        :return: dC/dinput
        """
        if input_node_idx == 0:     #   wrt Left matrix
            return self.total_cost_gradient @ self.input_nodes[1].output.T
        elif input_node_idx == 1:   #   wrt Right matrix
            return self.input_nodes[0].output.T @ self.total_cost_gradient
        raise ValueError("indices over 1 not applicable for {!r}".format(type(self).__name__))


class LogisticTransformNode(ParentNode):
    """
    Concrete subclass of ParentNode that transform input signal non-linearly using logistic function
    """
    def compute_output(self, input_signal: np.matrix) -> np.matrix:
        try:
            with np.errstate(over='raise'):
                return 1 / (1 + np.exp(input_signal))
        except FloatingPointError:
            # An element in input signal has caused result to be either 1 or 0 due to machine precision.
            # This causes problems upstream as this function mathematically should produce values in the range 0 < x < 1
            # => Change any any integer result to be non-integer by adding or subtracting the machine number epsilon
            with np.errstate(over='ignore'):
                result = 1 / (1 + np.exp(input_signal))
                result[result == 1.0] = 1 - eps
                result[result == 0.0] = eps
                return result

    def compute_cost_gradient_wrt_input_node(self, input_node_idx: int) -> np.matrix:
        return np.multiply(self.total_cost_gradient, (np.power(self.output, 2) - self.output))


class RootNode(ParentNode, MutableInputMixin):
    """
    Abstract implementation of ParentNode that can be considered a root or top node.
        This should be used as a singleton class
    """
    def post_init(self):
        super(RootNode, self).post_init()
        self.parent_nodes = tuple()
        self.touched = True
        self.do_backprop = True

    def compute_cost_gradient_wrt_input_node(self, input_node_idx: int):
        pass

    def compute_output(self, *inputs):
        pass


class TopNode(ParentNode):
    """
    Abstract implementation of ParentNode that can be considered a top node.
        It can have *no* parents except the singleton root node set at class definition
        Usually a cost or output node
        This Node has the method start_backprop which is the client facing method to back_propagate
    """
    root = RootNode(name=('root',))

    def post_init(self):
        super(TopNode, self).post_init()
        self.root += self
        self.parent_nodes = (self.root,)

    def start_backprop(self):
        self.backprop(np.matrix([[1]]), 'root')     # TODO fix this hack.  This node really shouldn't have a parent

    def check_grad(self, h=0.001, tol=.1):
        self.recur_reset()
        self.forward_prop()
        self.start_backprop()
        self.forward_prop()
        check_set = set()
        for inp in self.input_nodes_iter:
            inp.recur_check_grad(self, check_set, h=h, tol=tol)


class ConcatenateNode(ParentNode, MutableInputMixin):
    def __init__(self, *args, axis=0, **kwargs):
        self.axis = axis
        self.shapes = None
        super(ConcatenateNode, self).__init__(*args, **kwargs)

    def compute_output(self, *inputs):
        self.shapes = [inp.shape for inp in inputs]
        return np.concatenate(inputs, axis=self.axis)

    def compute_cost_gradient_wrt_input_node(self, input_node_idx: int):
        start = sum(shape[self.axis] for shape in self.shapes[:input_node_idx])
        length = self.shapes[input_node_idx][self.axis]
        _slice = slice(start, start+length)

        if self.axis == 0:
            return self.total_cost_gradient[_slice, :]
        else:
            return self.total_cost_gradient[:, _slice]


class EntropyCostNode(TopNode):
    def __init__(self, y, y_hat, name: tuple=tuple()):
        """
        Entropy function of comparing estimated output y_hat with ground truth output y
        :param
            y:      GroundTruthNode yielding true y
            y_hat:  output node that yield an estimate of y
        :return:
        """
        self.n = y.output.flatten().shape[1]
        super(EntropyCostNode, self).__init__(y, y_hat, name=name)

    def safe_log(self, vector):
        with np.errstate(divide='raise'):
            try:
                return np.log(vector)
            except ZeroDivisionError:
                with np.errstate(divide='ignore'):
                    result = np.log(vector)
                    result[vector == 1.0] = np.log(eps)
                    return result

    def compute_output(self, y, y_hat) -> np.matrix:
        """
        Cross entropy error function
            An eps is added to prevent the whole thing going haywire when we take the log of 0 in the case of y_hat
            elements being 0 or 1 precisely
        :param y_hat: Estimated y computed by forward prop of the network
        :param y: Ground Truth Node
        :return: Cross entropy error - np.matrix with shape = (1,1)
        """
        y_vec = np.matrix(y).reshape(self.n, 1)
        y_hat_vec = np.matrix(y_hat).reshape(self.n, 1)
        with np.errstate(divide='raise'):
            try:
                return - (1 / self.n) * (y_vec.T @ np.log(y_hat_vec) +
                                     ((1 - y_vec.T) @ np.log(1 - y_hat_vec)))
            except FloatingPointError:
                y_hat_vec[y_hat_vec==1.0] = 1 - eps
                y_hat_vec[y_hat_vec==0.0] = eps
                return - (1 / self.n) * (y_vec.T @ np.log(y_hat_vec) +
                                         ((1 - y_vec.T) @ np.log(1 - y_hat_vec)))

    def compute_cost_gradient_wrt_input_node(self, input_node_idx):
        if input_node_idx == 0:
            raise NotImplementedError()
        y = self.input_nodes[0].output
        y_hat = self.input_nodes[1].output
        return np.matrix((y_hat - y) / sigmoid_derivative(y_hat)) / self.n


class ResidualSumOfSquaresNode(TopNode):

    def __init__(self, y, y_hat, name: tuple=tuple()):
        """
        Entropy function of comparing estimated output y_hat with ground truth output y
        :param
            y:      GroundTruthNode yielding true y
            y_hat:  output node that yield an estimate of y
        :return:
        """
        self.n = len(y.output.flatten())
        super(ResidualSumOfSquaresNode, self).__init__(y, y_hat, name=name)

    def compute_output(self, y, y_hat) -> np.matrix:
        """
        Normed difference cost
            (y_hat - y).T @ (y_hat - y)
        :param y_hat: Estimated y computed by forward prop of the network
        :param y: Ground Truth Node
        :return: RSS
        """

        return (y.flatten() - y_hat.flatten()) @ (y.flatten() - y_hat.flatten()).T

    def compute_cost_gradient_wrt_input_node(self, input_node_idx) -> np.matrix:
        y = self.input_nodes[0].output
        y_hat = self.input_nodes[1].output
        return -2 * (y - y_hat)

root_node = TopNode.root
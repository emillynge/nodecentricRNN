__author__ = 'emil'
from . import nodes
from .utils import Theta, ThetaNodeGenerator, NodeName
import numpy as np
from collections import Sequence, defaultdict
from scipy import optimize

class SRNN(object):
    theta_generator = ThetaNodeGenerator


    minimizer = staticmethod(optimize.fmin_tnc)
    cost_node_class = nodes.EntropyCostNode
    output_link_class = nodes.LogisticTransformNode

    def __init__(self, inputs: np.matrix, outputs: np.matrix, *hidden_layer_sizes: Sequence):
        assert isinstance(inputs, np.matrix)
        assert isinstance(outputs, np.matrix)

        self.T = len(inputs)
        self.t_skips = self.T - len(outputs)

        self.inputs = inputs
        self.outputs = outputs
        self.n_out = outputs.shape[1]
        self.n_in = inputs.shape[1]
        self.theta = self.theta_generator(self.n_in, self.n_out, *hidden_layer_sizes)
        self.cost_node = None
        self.input_nodes = list()
        self.h = defaultdict(dict)  # Container for hidden vectors h_[t, l]
        self.est_output_node = None

    def add_t_step(self, t, input_vec, add_to_output=True):
        theta = self.theta
        x = nodes.InputNode(np.matrix(input_vec.reshape((self.n_in, 1))), name=NodeName('x', t=t))
        self.input_nodes.append(x)
        self.h[t][-1] = x

        output_linear_sum = nodes.AdditionNode(name=NodeName('outsum', t=t))
        for l in range(theta.weights.n_hidden):
            def name(var_name):
                return NodeName(var_name, t=t, l=l)

            bias = theta('bias', l)
            direct = theta('direct', l) @ self.h[t][l - 1]
            context = theta('context', l) @ self.h[t - 1][l]
            self.h[t][l] = nodes.LogisticTransformNode(nodes.AdditionNode(bias, direct, context),
                                                              name=name('h'))
            if add_to_output:
                output_linear_sum += theta('output', l) @ self.h[t][l]

        if add_to_output:
            if self.output_link_class is None:
                y_hat = output_linear_sum
            else:
                y_hat = self.output_link_class(output_linear_sum, name=NodeName('ŷ', t=t))
            self.est_output_node += y_hat

    def init_net(self):
        for l, shape in enumerate(self.theta.weights.context_conn_shapes):
            self.h[-1][l] = nodes.InputNode(np.zeros((shape[0], 1)), name=NodeName('zero', t=-1, l=l))
        self.est_output_node = nodes.ConcatenateNode(name=('Ŷ',), axis=1)

    def make_train_net(self):
        self.init_net()
        ground_truth = nodes.GroundTruthNode(np.matrix(self.outputs.T), name=NodeName('Y'))
        self.cost_node = self.cost_node_class(ground_truth, self.est_output_node, name=NodeName('C'))
        for t, input_vec in enumerate(self.inputs):
            self.add_t_step(t, input_vec, add_to_output=(t >= self.t_skips))

    def opt_func(self, theta_vec):
        theta = self.theta
        theta.weights.update_vector(theta_vec)
        theta.gradients.vector[:] = 0
        self.cost_node.forward_prop()
        J = float(self.cost_node.output[0, 0])
        self.cost_node.start_backprop()
        gradient_vector = theta.gradients.vector.tolist()
        if np.isnan(J):
            J = -np.inf
            gradient_vector[:] = np.zeros(theta.gradients.vector.shape).tolist()
        return J, gradient_vector

    def train(self):
        self.make_train_net()
        opt = self.minimizer(self.opt_func, self.theta.weights.vector)
        self.theta.weights.update_vector(opt[0])
        self.cost_node.forward_prop()
        print(self.outputs, '\n', self.est_output_node.output.T)
        self.cost_node.start_backprop()
        print(np.matrix(self.opt_func(opt[0])[0]))

    def check_grad(self):
        self.cost_node.check_grad()
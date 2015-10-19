# nodecentricRNN
Recurrent Neural Network from a node perspective

The backbone is the nodes module. Here you will find the most used node types defined:
* InputNode -> outputs the matrix that was passed at node creation
* ParameterNode -> same as InputNode, but also copies dCost/dParameter to matrix passed at node creation.
* GroundTruthNode -> same as InputNode, but simple another name to make debugging easier
* AdditionNode -> output sum its input nodes
* DotNode -> output dotproduct of first input node and second input node
* LogisticTransformNode -> output the logistic transform of its input node
* EntropyCostNode -> outputs the Entropy error of groundTruth Input and estimated input
* ConcatenateNode -> outputs all input nodes concatednated along a specified axis

# A simple network:
*pseudocode*
```
x1 = someinputmatrixattime1
x2 = someinputmatrixattime2

y = outputattime1and2

h0 = InputNodes(zeros)
theta_input = ParameterNode(weights_input, gradientmatrix_input)
theta_prev = ParameterNode(weights_prev, gradientmatrix_prev)
theta_out = ParameterNode(weights_out, gradientmatrix_out)

h1 = LogisticTranformNode(AdditionNode(DotNode(InputNode(x1), theta_input), DotNode(h0, theta_prev)))
y_hat1 = LogisticTranformNode(DotNode(h1, theta_out))

h2 = LogisticTranformNode(AdditionNode(DotNode(InputNode(x2), theta_input), DotNode(h1, theta_prev)))
y_hat2 = LogisticTranformNode(DotNode(h2, theta_out))

y_hat_vector = ConcatenateNode(y_hat1, y_hat2)
cost = EntropyCostNode(y, y_hat_vector)
```

cost can be computed by forward propagating and fetching output of cost:
```
cost.forward_prop()
C = cost.output
```

cost gradients can be obtained by backpropagating (requires forward propagation has been dones first). The resulting gradients will be stored in the gradient matrices supplied at declaration of parameter nodes
```
cost.start_backprop()
dCost_dthetainput = gradientmatrix_input
```

Use these gradients to update the weights be directly changing the weight matrices passed at declaration of parameter nodes.


`theta_input += gradient2updatestep(dCost_dthetainput)`

recompute and get new y estimate by forward propagating from concatenation node
```
y_hat_vector.forward_prop
new_estimate = y_hat_vector.output
```

## Infix Shorthand
To make coding easier to understand some infix operators are baked in at object creation:
* @: `node1 @ node2 -> DotNode(node1, node2)`
* +: `node1 + node2 -> Addition(node1, node2)` __if node1 is a concatenation node ->__ `ConcatenateNode(node2, *node1.input_nodes)`
* +=: `node1 += node2 -> node1.append(node2)` __if node1 is a concatenation node the effect is the same__

Using these we can refactor the simple network

```
x1 = someinputmatrixattime1
x2 = someinputmatrixattime2

y = GroundTruthNode(outputattime1and2)
y_hat = ConcatenateNode()

h0 = InputNode(zeros)
theta_input = ParameterNode(weights_input, gradientmatrix_input)
theta_prev = ParameterNode(weights_prev, gradientmatrix_prev)
theta_out = ParameterNode(weights_out, gradientmatrix_out)

h1 = LogisticTranformNode((InputNode(x1) @ theta_input) + (h0 @ theta_prev))
y_hat += LogisticTranformNode(h1 @ theta_out)

h2 = LogisticTranformNode((InputNode(x2) @ theta_input) + (h1 @ theta_prev))
y_hat += LogisticTranformNode(h2 @ theta_out)

cost = EntropyCostNode(y, y_hat)
```
## Node Naming
To make debugging easier it is possible to provide at name key-word when creating a node. This name should be a tuple with (name, level, timestep). In utils you will find the class NodeName which is a namedtuple that makes naming a bit easier. Naming the node makes the __repr__ function use this name instead of the usual representation.
```
print(LogisticTranformNode((InputNode(x1) @ theta_input) + (h0 @ theta_prev)).__repr__())
>>>  LogisticTranformNode(AdditionNode(DotNode(InputNode(), ParameterNode()), DotNode(InputNode(), ParameterNode())))
print(LogisticTranformNode((InputNode(x1, name=('x', 1)) @ theta_input) + (h0 @ theta_prev)).__repr__())
>>>  LogisticTranformNode(AdditionNode(DotNode(x[1], ParameterNode()), DotNode(InputNode(), ParameterNode())))
print(LogisticTranformNode((InputNode(x1) @ theta_input) + (h0 @ theta_prev), name=NodeName('h', t=1 , l=1).__repr__())
>>> h[1,1]
```

# Making your own nodes

To make a new node you should inherit from one of the node ABC's
* ParentNode: for nodes that has one or more inputnodes
* Topnode: for nodes that does not have any parent itself. mostly for new cost nodes
* Node: the most basic one. use only if above ABC's are not applicable 

Additionally there is a mutable input mixin that makes it possible to add or remove input nodes after object creation.
This mixin is used for example in AdditionNode and ConcatenateNode

Any node just has implement the abstract methods `compute_cost_gradient_wrt_input_node` and  `compute_output`
These are essential to doing the forward and backward propagation.

To check your implementation any topnode has the method `check_grad(self, h=0.001, tol=.1)` which will throw an error if any grandchild reports a gradient that differs too much from the central approximated gradient. *NB: This check will _only_ be done for ancestors to parameter nodes.*. 

# Recipes

recipes are ways of putting together nodes to form fully fledged netowrks and can be found in the `recipe` module. At the time of writing only a very simple SRNN recipe is there and no ABC for creating new a new recipe has been made.







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







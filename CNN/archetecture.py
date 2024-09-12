'''
This is were I will design the layers for the neural network, and the value-type for the layer.

All of the operations will be done with matrices operations, and I'll be preforming them with NumPy (for now).

Our class will create a computational graph for all the tensors, and it will go 'backward' from loss to all the nodes until we reach the parameters, or leaf, nodes. 
The way this will work, is that every single operation done on our matrices will be stored in the resulting value (parent matrix). We start at the 
final value, and go through all of the values that were a 'result' of the other matrices, and find their respective derivatives. We won't support
general cases like broadcasting, but we'll have enough flexibility to create custom network architectures.

Each layer type will have its own class:
- Linear layer
- Convolutional Layer
- Normalized layer (batchnorm)
- Tanh layer
- Softmax Layer (output)

Each layer will be able to feed into the other layers.
Derivations will be done on the level of layers.

'''

class AwesomeMatrix:

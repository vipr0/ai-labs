from numpy import array, random
from neurolib import NeuronLayer
from neurolib import NeuralNetwork
 
#Seed the random number generator
random.seed(1)

# Create layer 1 (3 neurons, each with 2 inputs)
layer1 = NeuronLayer(3, 2)

# Create layer 2 (a single neuron with 3 inputs)
layer2 = NeuronLayer(1, 3)

# Combine the layers to create a neural network
neural_network = NeuralNetwork(layer1, layer2)

print("Stage 1. Random starting synaptic weights: ")
neural_network.print_weights()

# The training set.
training_set_inputs = array([[1, 1], [1, 2], [2, 2], [0, 2]])
training_set_outputs = array([[0.2, 0.3, 0.4, 0.2]]).T

neural_network.train(training_set_inputs, training_set_outputs, 1000, 0.1)

# Export weights to file
neural_network.export_weights('weights.npz')

print("Stage 2. New synaptic weights after training: ")
neural_network.print_weights()
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

# Import weights from file
neural_network.import_weights('weights.npz')

# Input testing values
input_string = input('Enter input values (separated by comma): ')
input_array = [int(i) for i in input_string.split(',')]

# Test the neural network.
print(f"Testing inputs: {input_array} -> ?: ")
hidden_state, output = neural_network.think(array(input_array))
print(output)
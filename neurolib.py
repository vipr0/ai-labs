from numpy import exp, array, random, dot, all, savez, load


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
        self.output = 0
        self.delta = 0

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2


    def __sigmoid(self, x):
        '''
            The Sigmoid function, which describes an S shaped curve.
            We pass the weighted sum of the inputs through this function to
            normalise them between 0 and 1.
        ''' 
        return 1 / (1 + exp(-x))


    def __sigmoid_derivative(self, x):
        '''
            The derivative of the Sigmoid function.
            This is the gradient of the Sigmoid curve.
            It indicates how confident we are about the existing weight.
        '''
        return x * (1 - x)


    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, expected_delta):
        '''
            We train the neural network through a process of trial and error.
            Adjusting the synaptic weights each time.
        '''
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate delta
            delta = abs((output_from_layer_2 - training_set_outputs) / training_set_outputs)

            # Print result
            self.print_result(output_from_layer_2, training_set_outputs, delta)

            if(all(delta < expected_delta)):
                continue
                
            # Calculate the error for layer 2
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))        
        return output_from_layer1, output_from_layer2


    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 : ")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 :")
        print(self.layer2.synaptic_weights)
    

    def print_result(self, calculated_output, expected_output, delta):
        print(f'#:  Calculated Y:  Expected Y:  Delta:')
        for index, output in enumerate(expected_output):
            print(f'{index + 1}  {str(calculated_output[index]).center(13)}{str(output).center(13)}{delta[index]}')
    
    def export_weights(self, filename):
        open(filename, 'w').close()
        savez(
            filename, 
            w1 = self.layer1.synaptic_weights,
            w2 = self.layer2.synaptic_weights
            )

    def import_weights(self, filename):
        weights_file = load(filename)
        self.layer1.synaptic_weights = weights_file['w1']
        self.layer2.synaptic_weights = weights_file['w2']

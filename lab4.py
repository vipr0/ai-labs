import pandas as pd
from neurolib import HiddenNeuron as hn
from neurolib import OutputNeuron as on

permissible_delta = 0.1
expected_y = 0.3
x = 1
w = [[0.2, -0.5]]

y = []
deltas = []

running = True

while running:
    # calculate hidden neuron
    hidden_neuron = hn(w = [w[-1][0]], x = [x])

    # calculate output neuron
    output_neuron = on(
        w = [w[-1][1]], 
        x = [hidden_neuron.y], 
        exp_y = expected_y
        )
    if output_neuron.delta <= permissible_delta:
        running = False
    else:
        # correction for output neuron
        output_delta_w = output_neuron.calculate_delta_w()

        # correction for hidden neuron
        hidden_delta_w = hidden_neuron.calculate_delta_w(
            next_sigma = output_neuron.sigma,
            w = w[-1][1]
        )

        # append new values
        w.append(
            [
                round(w[-1][0] + hidden_delta_w[0],3),
                round(w[-1][1] + output_delta_w[0], 3)
            ]
        )
        y.append([hidden_neuron.y, output_neuron.y])
        deltas.append(output_neuron.delta)

results_dict = {
    'Delta': deltas,
    'Y1, Y2': y,
    'W1, W2': w[1:]
}

results = pd.DataFrame(results_dict)
print(results)
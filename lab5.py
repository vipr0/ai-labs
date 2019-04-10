import math
import pandas as pd
import numpy as np
from neurolib import HiddenNeuron as hn
from neurolib import OutputNeuron as on

permissible_delta = 0.1
expected_y = [0.2, 0.3, 0.3, 0.35]

w_array_1 = [np.random.rand(3, 2)]
w_array_2 = [np.random.rand(3)]
x = [[0.1, 0.1], [0.2, 0.1], [0.15, 0.15], [0.1, 0.25]]

y = [[0, 0, 0, 0]]
deltas = []

running = True
iteration = range(len(x))
w_final_1 = [0, 0, 0, 0]
w_final_2 = [0, 0, 0, 0]

while running:
    y.append([]) # creating new cell for results
    for index in iteration:
        # creating 1st hidden neuron
        hidden_1 = hn(w=w_array_1[-1][0], x=x[index])

        # creating 2nd hidden neuron
        hidden_2 = hn(w=w_array_1[-1][1], x=x[index])

        # creating 3rd hidden neuron
        hidden_3 = hn(w=w_array_1[-1][2], x=x[index])

        # creating output neuron
        output = on(
            w=w_array_2[-1],
            x=[hidden_1.y, hidden_2.y, hidden_3.y],
            exp_y=expected_y[index])

        if output.delta > permissible_delta:
            # correction for output neuron
            new_w_array_2 = [round(elem + output.delta_w[index], 3)
                             for index, elem in enumerate(w_array_2[-1])]

            # delta of W for hidden neurons
            delta_w_1 = hidden_1.calculate_delta_w(
                output.sigma, w_array_2[-1][0])
            delta_w_2 = hidden_1.calculate_delta_w(
                output.sigma, w_array_2[-1][1])
            delta_w_3 = hidden_1.calculate_delta_w(
                output.sigma, w_array_2[-1][2])

            # add new W for hidden neurons
            w_array_1.append([[round(elem + delta_w_1[index], 3)
                               for index, elem in enumerate(w_array_1[-1][0])],
                              [round(elem + delta_w_2[index], 3)
                               for index, elem in enumerate(w_array_1[-1][1])],
                              [round(elem + delta_w_3[index], 3)
                               for index, elem in enumerate(w_array_1[-1][2])]])

            # add new W for output neurons
            w_array_2.append(new_w_array_2)

            y[-1].append(output.y)

            deltas.append(output.delta)
        else:
            y[-1].append(output.y)
            running = False

results = {
    'w_1': w_array_1,
    'w_2': w_array_2,
}

pd_results = pd.DataFrame(results)

print(pd_results)
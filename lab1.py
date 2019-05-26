from math import e
import numpy as np

print('Lab #1')

number_of_inputs = int(input('Enter number of inputs: '))
alpha = float(input('Enter alpha: '))
x = np.array([]) # list of signals
w = np.array([]) # list of multipliers

# filling values of signals and multipliers
for i in range(number_of_inputs):
    x = np.append(x, float(input(f'Enter x{i}: ')))
    w = np.append(w, float(input(f'Enter w{i}: ')))

# y = 1 / (1 + e ** (- alpha * np.dot(x, w)))
y = 1 / (1 + np.exp(-alpha * np.dot(x, w)))

print(f' y = {y}')
from math import e

print('Lab #1')

number_of_inputs = int(input('Enter number of inputs: '))
alpha = float(input('Enter alpha: '))
signals = [] # list of signals and multipliers

# filling values of signals and multipliers
for i in range(number_of_inputs):
    x = float(input(f'Enter x{i}: '))
    w = float(input(f'Enter w{i}: '))
    signals.append((x, w))

sum_of_signals = 0
for x, w in signals:
    sum_of_signals += x * w

y = 1 / (1 + e ** (- alpha * sum_of_signals))

print(f' y = {y}')
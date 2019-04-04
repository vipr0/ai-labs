import math
import pandas as pd

permissible_delta = 0.1
alpha = 1
expected_y = 0.7
x = 1

i = 1

w = [0.1]
y = []

def get_sum_of_x(x, w):
    return x * w

def calculate_y(sum_of_x):
    y.append(1 / (1 + math.e ** (- alpha * sum_of_x)))
    return y[-1]

def calculate_delta(current_y):
    return abs((current_y - expected_y) /  expected_y)

while True:
    sum_of_x = get_sum_of_x(x, w[i - 1])
    current_y = calculate_y(sum_of_x)
    current_delta = calculate_delta(current_y)

    if current_delta <= permissible_delta:
        break
    else:
        current_sigma = current_y * (1 - current_y) * (expected_y - current_y)
        delta_w = x * current_sigma
        w.append(w[i -1] + delta_w)
        i += 1

results_dict = {
    'w': w,
    'y': y
}

results = pd.DataFrame(results_dict)
print(results)
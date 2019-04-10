from math import e


class Neuron:
    """
    A class used to represent an Neuron

    ...

    Attributes
    ----------
    w : list
        a list of input signals multipliers
    x : list
        a list of input signals
    alpha : float
        (default 1)

    Methods
    -------
    get_sum_of_inputs()
        Returns sum of all input signals multiplied by multiplier 
    calculate_y()
        Returns output signal
    """

    def __init__(self, w, x, alpha=1):
        self.w = w
        self.x = x
        self.alpha = alpha
        self.sum_inputs = self.get_sum_of_inputs()
        self.y = self.calculate_y()

    def get_sum_of_inputs(self):
        """Returns sum of all input signals multiplied by multiplier"""
        return sum([elem * self.w[index] for index, elem in enumerate(self.x)])

    def calculate_y(self):
        """Returns output signal"""
        return 1 / (1 + e ** (- self.alpha * self.get_sum_of_inputs()))
    
    def set_w(self, w):
        """Set new W value for neuron"""
        self.w = w

    def set_x(self, x):
        """Set new input value for neuron"""
        self.x = x

class HiddenNeuron(Neuron):
    """
    A class used to represent an Hidden Neuron

    ...

    Attributes
    ----------
    w : list
        a list of input signals multipliers
    x : list
        a list of input signals
    alpha : float
        (default 1)

    Methods
    -------
    calculate_sigma(sigma = , w = )
        Returns sigma for current neuron
    calculate_delta_w(next_sigma, w)
        Returns list of deltas of W for current neuron inputs.
    """

    def calculate_sigma(self, sigma, w):
        """Returns sigma for current neuron.

        Parameters
        ----------
        sigma : float
            Sigma from next neuron
        w : float
            Previous W from next neuron

        """
        return self.y * (1 - self.y) * (sigma * w)

    def calculate_delta_w(self, next_sigma, w):
        """Returns list of deltas of W for current neuron inputs.

        next_sigma and w need to calculate sigma

        Parameters
        ----------
        next_sigma : float
            Sigma from next neuron
        w : float
            Previous W from next neuron

        """
        sigma = self.calculate_sigma(next_sigma, w)
        return [elem * sigma for elem in self.x]


class OutputNeuron(Neuron):
    """
    A class used to represent an Output Neuron

    ...

    Attributes
    ----------
    w : list
        a list of input signals multipliers
    x : list
        a list of input signals
    alpha : float
        (default 1)
    exp_y : float
        an expected output for network

    Methods
    -------
    calculate_delta()
        Returns delta of neuron
    calculate_sigma()
        Returns sigma for current neuron
    calculate_delta_w()
        Returns list of deltas of W for current neuron inputs.
    """

    def __init__(self, w, x, exp_y):
        Neuron.__init__(self, w, x)
        self.exp_y = exp_y
        self.delta = self.calculate_delta()
        self.sigma = self.calculate_sigma()
        self.delta_w = self.calculate_delta_w()

    def calculate_delta(self):
        """Returns delta of neuron"""
        return abs((self.y - self.exp_y) / self.exp_y)

    def calculate_sigma(self):
        """Returns sigma for current neuron."""
        return self.y * (1 - self.y) * (self.exp_y - self.y)

    def calculate_delta_w(self):
        """Returns list of deltas of W for current neuron inputs."""
        return [elem * self.calculate_sigma() for elem in self.x]

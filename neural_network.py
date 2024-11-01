# neural_network

from typing import List, Callable
from math import exp

ActivationFunc = Callable[[float], float]

def logistic_func(x: float) -> float:
    """
    Logistic activation function.
    """
    return 1.0 / (1.0 + exp(-x))

class Neuron:
    """
    Represents a neuron in a neural network.
    """
    def __init__(self, weights: List[float]) -> None:
        """
        Initializes the neuron instance with the given weights.
        """
        self.weights = weights
    
    def propagate(self, inputs: List[float], f: ActivationFunc = logistic_func) -> float:
        """
        Propagates the inputs through the neuron and returns the output.
        """
        if len(inputs) != len(self.weights) - 1:
            raise ValueError("The number of inputs must be equal to the number of weights minus 1 (the bias).")
        inputs = [1] + inputs # add a neutral weight for the bias for easy computation
        return f(sum([i * w for i, w in zip(inputs, self.weights)]))

if __name__ == "__main__":
    def test_neuron():
        print("============ TESTING NEURON ============")
        weights = [0.5, -0.6, 0.1]
        inputs = [1.0, 2.0]
        neuron = Neuron(weights)
        
        # Test with logistic function
        output_logistic = neuron.propagate(inputs, logistic_func)
        expected_logistic = logistic_func(0.5 * 1 + -0.6 * 1.0 + 0.1 * 2.0)
        assert abs(output_logistic - expected_logistic) < 1e-6, f"Expected {expected_logistic}, got {output_logistic}"

        print("All tests passed.")
        print("========================================")
    
    print("Running unit tests...")
    test_neuron()
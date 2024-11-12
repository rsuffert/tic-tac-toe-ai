import random
from typing import List, Tuple
from neural_network import NeuralNetwork


def generateRandomWeights(topology: Tuple[int, ...]) -> List[float]:
    """
    Generates a list of random weights for a neural network with the given topology.
    """
    weights = []
    for prev_layer_size, cur_layer_size in zip(topology[:-1], topology[1:]):
        synapses_per_neuron = prev_layer_size + 1  # +1 for bias
        weights.extend([random.uniform(-1.0, 1.0)
                       for _ in range(synapses_per_neuron * cur_layer_size)])
    return weights


class Population:
    def __init__(self, size: int, topology: Tuple[int, ...]) -> None:
        self.size = size
        self.individuals: List[NeuralNetwork] = [
            NeuralNetwork(topology=topology, weights=generateRandomWeights(topology)) for _ in range(size)
        ]

    def getPopulationSize(self):
        return self.size

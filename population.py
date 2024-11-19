import random
from typing import List, Tuple
from neural_network import NeuralNetwork
from aiplayers import Difficulty
from fitness import fitness


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
        self.topology = topology
        self.individuals: List[List[float]] = [
            generateRandomWeights(topology) + [0.0] for _ in range(size)
        ]

    def getPopulationSize(self) -> int:
        return self.size

    def getIndividual(self, index: int) -> List[float]:
        """
        Returns the weights for the individual at the given index.
        """
        return self.individuals[index]

    def setIndividual(self, index: int, weights: List[float]) -> None:
        """
        Sets the weights for the individual at the given index.
        """
        self.individuals[index] = weights

    def updateFitness(self, difficulty: Difficulty) -> None:
        """
        Updates the fitness for all individuals in the population.
        """
        for i in range(self.size):
            weights = self.individuals[i][:-1]
            fitness_value = fitness(weights, difficulty, self.topology)
            self.individuals[i][-1] = fitness_value

    def sortIndividualsByFitness(self) -> None:
        """
        Sorts the individuals by their fitness in descending order.
        """
        self.individuals.sort(key=lambda ind: ind[-1], reverse=True)

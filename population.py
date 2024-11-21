import random
from typing import List, Tuple
from neural_network import NeuralNetwork
from aiplayers import Difficulty, MinimaxAIPlayer, Player
from neural_network import NeuralNetwork
from tictactoeutils import Classifier

classifier: Classifier = Classifier(
    Player.X.value, Player.O.value, Player.EMPTY.value)

minimax: MinimaxAIPlayer = MinimaxAIPlayer()


def fitness(individual: List[float], difficulty: Difficulty, NN_TOPOLOGY: Tuple[int, ...]) -> float:
    """
    Calculates the fitness of an individual of a population.
    Args:
        individual (List[float]): The weights to be used in the neural network.
        difficulty (Difficulty): The difficulty level to which the individual will be evaluated against.
    Returns:
        float: A number representing the fitness of the individual. The higher the number, the better the individual.
    """
    fitness: float = 0.0
    board: List[List[Player]] = [
        [Player.EMPTY for _ in range(3)] for _ in range(3)]
    network: NeuralNetwork = NeuralNetwork(NN_TOPOLOGY, individual)

    # simulate a game between the neural network and the minimax AI
    result: str = classifier.ONGOING
    while result == classifier.ONGOING:
        # neural network plays as X
        row, col = network.predict(_to_float_board(board))
        if board[row][col] != Player.EMPTY:
            # terminate the game if the network makes an invalid move (but still value how far it got)
            return fitness
        fitness += 1.0  # grant one point for each valid move by the network
        board[row][col] = Player.X

        # classify the board after the neural network plays
        result = classifier.classify(_board_to_string(board))
        if result != classifier.ONGOING:
            break

        # minimax plays as O
        board = minimax.play(difficulty, board)

        # classify the board after minimax plays
        result = classifier.classify(_board_to_string(board))

    # grant bonus or penalty based on the result of the match
    match result:
        # grant 25% bonus if the neural network wins
        case classifier.X_WON: fitness *= 1.25
        case classifier.O_WON: fitness *= 0.75  # penalize 25% if minimax wins
        case classifier.TIE: fitness *= 1.12  # 12% bonus for a tie

    return fitness


def _board_to_string(board: List[List[Player]]) -> str:
    """Converts the board to a string."""
    return ','.join(str(cell.value) for row in board for cell in row)


def _to_float_board(board: List[List[Player]]) -> List[List[float]]:
    """Converts the board to a list of list of floats."""
    return [[float(cell.value) for cell in row] for row in board]


def generateRandomWeights(topology: Tuple[int, ...]) -> List[float]:
    """
    Generates a list of random weights for a neural network with the given topology,
    including bias weights.
    """
    weights = []
    for prev_layer_size, cur_layer_size in zip(topology[:-1], topology[1:]):
        # Generate weights for each neuron in the current layer, including bias
        weights.extend([random.uniform(-1.0, 1.0)
                       for _ in range((prev_layer_size + 1) * cur_layer_size)])
    return weights


class Population:
    def __init__(self, size: int, topology: Tuple[int, ...]) -> None:
        self.size = size
        self.topology = topology
        self.individuals: List[List[float]] = [
            generateRandomWeights(topology) + [0.0] for _ in range(size)
        ]

    def clear(self) -> None:
        self.individuals = []

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

    def updateFitness(self, difficulty: Difficulty, elitism: bool = False) -> None:
        """
        Updates the fitness for all individuals in the population.
        """
        print('Initializing fitness values...')
        start_index = 1 if elitism else 0
        for i in range(start_index, self.size):
            weights = self.individuals[i][:-1]
            fitness_value = fitness(weights, difficulty, self.topology)
            self.individuals[i][-1] = fitness_value
        print('Fitness values initialized.')

    def sortIndividualsByFitness(self) -> None:
        """
        Sorts the individuals by their fitness in descending order.
        """
        self.individuals.sort(key=lambda ind: ind[-1], reverse=True)

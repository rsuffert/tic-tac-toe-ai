# neural_network.py
import json
from math import exp
from typing import Callable, List, Tuple
from aiplayers import Player, Difficulty, MinimaxAIPlayer
from tictactoeutils import Classifier


classifier: Classifier = Classifier(
    Player.X.value, Player.O.value, Player.EMPTY.value)

ActivationFunc = Callable[[float], float]


minimax: MinimaxAIPlayer = MinimaxAIPlayer()


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
            raise ValueError(
                "The number of inputs must be equal to the number of weights minus 1 (the bias)."
            )
        # add a neutral weight for the bias for easy computation
        inputs = [1] + inputs
        return f(sum([i * w for i, w in zip(inputs, self.weights)]))


class NeuralNetwork:
    """
    Class NeuralNetwork
    network: Representation of the neural network. Each sublist represents a layer of the network and each element of a sublist is a neuron of the respective layer.
    """

    def __init__(self, topology: Tuple[int, ...], weights: List[float]) -> None:
        """
        Initializes a fully connected neural network with the given topology
        (including input and output layers) and weights.
        """
        self._validate(topology, weights)
        self.network: List[List[Neuron]] = []
        self.weights = weights

        used_weights: int = 0
        for prev_layer_size, cur_layer_size in zip(topology[:-1], topology[1:]):
            synapses_per_neuron: int = prev_layer_size + 1  # + 1 for bias
            layer: List[Neuron] = [
                Neuron(
                    weights[
                        used_weights + n * synapses_per_neuron: used_weights
                        + (n + 1) * synapses_per_neuron
                    ]
                )
                for n in range(cur_layer_size)
            ]
            used_weights += synapses_per_neuron * cur_layer_size
            self.network.append(layer)

    @staticmethod
    def _validate(topology: Tuple[int, ...], weights: List[float]) -> None:
        """
        Validates the topology and weights of the network, making sure the number of weights matches
        the number of synapses in the network. It raises a ValueError if the validation fails.
        """
        if len(topology) < 2:
            raise ValueError(
                "The topology must have at least two layers (input and output)."
            )

        synapses_per_layer: List[int] = [
            (topology[i] + 1) * topology[i + 1] for i in range(len(topology) - 1)
        ]

        if sum(synapses_per_layer) != len(weights):
            raise ValueError(
                f"Expected {sum(synapses_per_layer)} weights, got {
                    len(weights)}."
            )

    def predict(self, board: List[List[float]]) -> Tuple[int, int]:
        inputs = [cell for row in board for cell in row]
        for layer in self.network:
            outputs = [neuron.propagate(inputs) for neuron in layer]
            inputs = outputs
        max_index = outputs.index(max(outputs))
        return divmod(max_index, len(board))

    def fitness(individual: List[float], difficulty: Difficulty) -> float:
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
            case classifier.TIE: pass            # no bonus or penalty for a tie

        return fitness

    def to_file(self, file_path: str) -> None:
        model = {
            "topology": [len(self.network[0][0].weights) - 1]
            + [len(layer) for layer in self.network],
            "weights": [
                weight
                for layer in self.network
                for neuron in layer
                for weight in neuron.weights
            ],
        }
        with open(file_path, "w") as f:
            json.dump(model, f)

    @classmethod
    def from_file(cls, file_path: str) -> "NeuralNetwork":
        with open(file_path, "r") as f:
            model = json.load(f)
        topology = tuple(model["topology"])
        weights = model["weights"]
        return cls(topology, weights)


@classmethod
def _board_to_string(board: List[List[Player]]) -> str:
    """Converts the board to a string."""
    return ','.join(str(cell.value) for row in board for cell in row)


@classmethod
def _to_float_board(board: List[List[Player]]) -> List[List[float]]:
    """Converts the board to a list of list of floats."""
    return [[float(cell.value) for cell in row] for row in board]

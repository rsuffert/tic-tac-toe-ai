import json
from math import exp
from typing import Callable, List, Tuple  # neural_network

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

    def propagate(
        self, inputs: List[float], f: ActivationFunc = logistic_func
    ) -> float:
        """
        Propagates the inputs through the neuron and returns the output.
        """
        if len(inputs) != len(self.weights) - 1:
            raise ValueError(
                "The number of inputs must be equal to the number of weights minus 1 (the bias)."
            )
        inputs = [1] + inputs  # add a neutral weight for the bias for easy computation
        return f(sum([i * w for i, w in zip(inputs, self.weights)]))


def test_neuron():
    print("============ TESTING NEURON ============")
    weights = [0.5, -0.6, 0.1]
    inputs = [1.0, 2.0]
    neuron = Neuron(weights)

    # Test with logistic function
    output_logistic = neuron.propagate(inputs, logistic_func)
    expected_logistic = logistic_func(0.5 * 1 + -0.6 * 1.0 + 0.1 * 2.0)
    assert (
        abs(output_logistic - expected_logistic) < 1e-6
    ), f"Expected {expected_logistic}, got {output_logistic}"

    print("All tests passed.")
    print("========================================")


class NeuralNetwork:
    """
    Classe NeuralNetwork
    network: Representação da rede neural. Cada sub-lista representa uma camada da rede e cada elemento de uma sub-lista é um neurônio da respectiva camada.
    """

    def __init__(
        self, topology: Tuple[int, ...], weights: List[List[List[float]]]
    ) -> None:
        """
        Construtor de rede neural, que recebe uma tupla de inteiros que representa a topologia da rede desejada e os pesos a serem aplicados aos neurônios.
        """
        self.network = []
        for layer_size, layer_weights in zip(topology, weights):
            layer = [Neuron(neuron_weights) for neuron_weights in layer_weights]
            self.network.append(layer)

    def predict(self, board: List[List[float]]) -> Tuple[int, int]:
        """
        Recebe uma instância de tabuleiro e retorna uma tupla representando a linha e coluna na qual a rede está jogando.
        """
        inputs = [cell for row in board for cell in row]
        for layer in self.network:
            outputs = [neuron.propagate(inputs) for neuron in layer]
            inputs = outputs
        # Assuming the output is a flat list of probabilities for each cell
        max_index = outputs.index(max(outputs))
        return divmod(max_index, len(board))

    def to_file(self, file_path: str) -> None:
        """
        Salva o modelo em um arquivo.
        """
        model = {
            "topology": [len(layer) for layer in self.network],
            "weights": [
                [[weight for weight in neuron.weights] for neuron in layer]
                for layer in self.network
            ],
        }
        with open(file_path, "w") as f:
            json.dump(model, f)

    @classmethod
    def from_file(cls, file_path: str) -> "NeuralNetwork":
        """
        Lê o modelo e instancia a rede.
        """
        with open(file_path, "r") as f:
            model = json.load(f)
        topology = tuple(model["topology"])
        weights = model["weights"]
        return cls(topology, weights)


if __name__ == "__main__":
    print("Running unit tests...")
    test_neuron()

    topology = (9, 5, 9)
    weights = [
        [
            [0.1] * 10 for _ in range(5)
        ],  # 5 neurons, each with 10 weights (9 inputs + 1 bias)
        [
            [0.1] * 6 for _ in range(9)
        ],  # 9 neurons, each with 6 weights (5 inputs + 1 bias)
    ]
    nn = NeuralNetwork(topology, weights)
    board = [[0.0] * 3 for _ in range(3)]  # Example board
    print(nn.predict(board))
    nn.to_file("model.json")
    nn_loaded = NeuralNetwork.from_file("model.json")
    print(nn_loaded.predict(board))

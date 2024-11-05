# neural_network
import json
from math import exp
from typing import Callable, List, Tuple

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

        used_weights: int = 0
        for prev_layer_size, cur_layer_size in zip(topology[:-1], topology[1:]):
            synapses_per_neuron: int = prev_layer_size + 1  # + 1 for bias
            layer: List[Neuron] = [
                Neuron(
                    weights[
                        used_weights + n * synapses_per_neuron : used_weights
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
                f"Expected {sum(synapses_per_layer)} weights, got {len(weights)}."
            )

    def predict(self, board: List[List[float]]) -> Tuple[int, int]:
        inputs = [cell for row in board for cell in row]
        for layer in self.network:
            outputs = [neuron.propagate(inputs) for neuron in layer]
            inputs = outputs
        max_index = outputs.index(max(outputs))
        return divmod(max_index, len(board))

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


if __name__ == "__main__":
    print("Running unit tests...")

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

    def test_neural_network():
        print("======== TESTING NEURAL NETWORK ========")
        topology = (9, 5, 9)
        weights = [0.1] * 104
        nn = NeuralNetwork(topology, weights)
        board = [[0.0] * 3 for _ in range(3)]

        # Test predict function
        prediction = nn.predict(board)
        assert (
            isinstance(prediction, tuple) and len(prediction) == 2
        ), "Prediction should be a tuple of two integers."

        # Test serialization and deserialization
        nn.to_file("model.json")
        nn_loaded = NeuralNetwork.from_file("model.json")
        assert nn.predict(board) == nn_loaded.predict(
            board
        ), "Loaded model should produce the same prediction."

        print("All tests passed.")
        print("========================================")
    
    def test_neural_network_weights_assignment_topology_2_3_2():
        print("======== TESTING NEURAL NETWORK WEIGHTS ASSIGNMENT FOR TOPOLOGY (2, 3, 2) ========")
        
        topology = (2, 3, 2)
        weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        nn = NeuralNetwork(topology, weights)
        
        expected_weights = [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],  # First hidden layer
            [[1.0, 1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 1.7]]          # Output layer
        ]
        
        for layer_idx, layer in enumerate(nn.network):
            for neuron_idx, neuron in enumerate(layer):
                assert neuron.weights == expected_weights[layer_idx][neuron_idx], (
                    f"Expected weights {expected_weights[layer_idx][neuron_idx]}, "
                    f"got {neuron.weights} for neuron {neuron_idx} in layer {layer_idx}"
                )
        
        print("All tests passed.")
        print("========================================")

    test_neuron()
    test_neural_network()
    test_neural_network_weights_assignment_topology_2_3_2()

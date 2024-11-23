from neural_network import NeuralNetwork
import random
from typing import List, Tuple
from aiplayers import Difficulty, MinimaxAIPlayer, Player
from tictactoeutils import Classifier

classifier: Classifier = Classifier(Player.X.value, Player.O.value, Player.EMPTY.value)  # noqa

minimax: MinimaxAIPlayer = MinimaxAIPlayer()


class Population:
    def __init__(self, size: int, topology: Tuple[int, ...]) -> None:
        self.size = size
        self.topology = topology
        self.individuals: List[List[float]] = [generateRandomWeights(topology) + [0.0] for _ in range(size)]  # noqa

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

    def updateElitism(self) -> None:
        """
        Updates the elitism of the population.
        """

        max = -1
        max_index = 0

        for i in range(0, self.size):
            if self.individuals[i][-1] > max:
                max = self.individuals[i][-1]
                max_index = i

        best_individual = self.individuals[max_index]
        self.individuals[0] = best_individual


class GeneticAlgorithm:
    def __init__(self, population_size: int, topology: Tuple[int, ...], crossover_rate: float, mutation_rate: float, max_generations: int, elitism: bool):
        self.population_size = population_size
        self.topology = topology
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.elitism = elitism
        self.population = Population(population_size, topology)

    def run(self):
        print("Running genetic algorithm...")
        difficulty = self.random_difficulty()
        self.population.updateFitness(difficulty, self.elitism)
        self.population.updateElitism()
        best_fitness = self.population.getIndividual(0)[-1]
        print(f"Generation 1: Best fitness = {best_fitness}, Difficulty = {difficulty.name}")  # noqa

        for generation in range(2, self.max_generations + 1):
            difficulty = self.random_difficulty()
            self.population = self.new_generation(self.population, difficulty)
            best_fitness = self.population.getIndividual(0)[-1]
            print(f"Generation {generation}: Best fitness = {best_fitness}, Difficulty = {difficulty.name}")  # noqa

    def new_generation(self, population: Population, difficulty: Difficulty) -> Population:
        new_population = Population(self.population_size, self.topology)
        new_population.clear()
        if self.elitism:
            new_population.individuals.append(population.getIndividual(0))

        while len(new_population.individuals) < self.population_size:
            parent1 = self.selection_tournament(population)
            parent2 = self.selection_tournament(population)
            offspring = self.crossover(parent1, parent2)

            for child in offspring:
                self.mutate(child)
                new_population.individuals.append(child)

        new_population.updateFitness(difficulty, self.elitism)
        new_population.updateElitism()
        return new_population

    def selection_tournament(self, population: Population) -> List[float]:
        tournament_size = 2
        selected = random.sample(population.individuals, tournament_size)
        selected.sort(key=lambda ind: ind[-1], reverse=True)
        return selected[0]

    def crossover(self, parent1: List[float], parent2: List[float]) -> List[List[float]]:
        child_weights = [(w1 + w2) / 2 for w1, w2 in zip(parent1[:-1], parent2[:-1])]  # noqa
        child_weights.append(0.0)  # Initialize fitness value for the child
        return [child_weights]

    def mutate(self, individual: List[float]):
        for i in range(len(individual) - 1):
            if random.random() <= self.mutation_rate:
                individual[i] = random.uniform(-1.0, 1.0)

    def random_difficulty(self) -> Difficulty:
        """
        Generates a random difficulty with the following probabilities:
        - 33% chance of being EASY
        - 33% chance of being MEDIUM
        - 33% chance of being HARD
        """
        rand = random.random()
        if rand < 0.3:
            return Difficulty.EASY
        elif rand < 0.66:
            return Difficulty.MEDIUM
        else:
            return Difficulty.HARD


def fitness(individual: List[float], difficulty: Difficulty, NN_TOPOLOGY: Tuple[int, ...]) -> float:
    """
    Calculates the fitness of an individual of a population.
    Args:
        individual (List[float]): The weights to be used in the neural network.
        difficulty (Difficulty): The difficulty level to which the individual will be evaluated against.
        NN_TOPOLOGY (Tuple[int, ...]): The topology of the neural network.
    Returns:
        float: A number representing the fitness of the individual. The higher the number, the better the individual.
    """
    fitness: float = 0.0
    board: List[List[Player]] = [[Player.EMPTY for _ in range(3)] for _ in range(3)]  # noqa
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


if __name__ == "__main__":
    population_size = 10
    topology = (9, 9, 9)
    crossover_rate = 0.9
    mutation_rate = 0.01
    max_generations = 15
    elitism = True

    ga = GeneticAlgorithm(population_size, topology,
                          crossover_rate, mutation_rate, max_generations, elitism)
    ga.run()

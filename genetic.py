import random
from typing import List, Tuple
from aiplayers import Difficulty, MinimaxAIPlayer, Player
from tictactoeutils import Classifier
from neural_network import NeuralNetwork
import concurrent.futures

classifier: Classifier = Classifier(Player.X.value, Player.O.value, Player.EMPTY.value)
minimax: MinimaxAIPlayer = MinimaxAIPlayer()

def fitness(individual: List[float], difficulty: Difficulty, nn_topology: Tuple[int, ...]) -> float:
    """
    Calculates the fitness of an individual of a population.
    Args:
        individual (List[float]): The weights to be used in the neural network.
        difficulty (Difficulty): The difficulty level to which the individual will be evaluated against.
        nn_topology (Tuple[int, ...]): The topology of the neural network.
    Returns:
        float: A number representing the fitness of the individual. The higher the number, the better the individual.
            The maximum fitness value possible is 6.25 (if the network plays all its turns correclty and, at the end,
            wins), and the minimum is 1.0, since it always begins playing, so at least one correct move will be made.
    """
    fitness: float = 0.0
    board: List[List[Player]] = [
        [Player.EMPTY for _ in range(3)] for _ in range(3)]
    network: NeuralNetwork = NeuralNetwork(nn_topology, individual)

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


class Population:
    def __init__(self, size: int, topology: Tuple[int, ...]) -> None:
        self.size = size
        self.topology = topology
        self.individuals: List[List[float]] = []
    
    def with_random_individuals(self) -> "Population":
        """
        Initializes the population with random individuals according to the topology specified in the constructor.
        """
        def generate_random_weights(topology: Tuple[int, ...]) -> List[float]:
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
            
        self.individuals = [
            generate_random_weights(topology) + [0.0] for _ in range(self.size)
        ]
        return self

    def update_fitness(self, difficulty: Difficulty) -> None:
        """Updates the fitness for all individuals in the population."""
        def eval_fitness(index: int) -> Tuple[int, float]:
            return index, fitness(
                self.individuals[index][:-1], 
                difficulty, 
                self.topology
            )

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(eval_fitness, i) for i in range(self.size)]
            for future in concurrent.futures.as_completed(futures):
                index, fit_val = future.result()
                self.individuals[index][-1] = fit_val
                print(f"\t{index+1}. Fitness = {fit_val}")

    def sort_individuals_by_fitness(self) -> None:
        """Sorts the individuals by their fitness in descending order."""
        self.individuals.sort(key=lambda ind: ind[-1], reverse=True)


class GeneticAlgorithm:
    def __init__(
        self, population_size: int, topology: Tuple[int, ...], mutation_rate: float,
        max_generations: int, elitism: bool
    ):
        self.population_size = population_size
        self.topology = topology
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.elitism = elitism
        self.population = Population(population_size, topology).with_random_individuals()

    def run(self):
        """Runs the genetic algorithm with the parameters specified in the constructor."""
        print("Training has started...")
        for g in range(self.max_generations):
            difficulty = self.random_difficulty()

            self.population = self.new_generation(self.population, difficulty)
            print(f"Generation {g+1} has been trained. Best fitness is {self.population.individuals[0][-1]} and difficulty is {difficulty}")
            
            if self.population.individuals[0][-1] >= 6.25:
                print("Elite individual has reached the maximum fitness. Stopping training...")
                break

    def new_generation(self, population: Population, difficulty: Difficulty) -> Population:
        """Takes in the current population and returns the next generation of individuals."""
        new_population = Population(self.population_size, self.topology)

        if self.elitism:
            population.sort_individuals_by_fitness()
            new_population.individuals.append(population.individuals[0])

        while len(new_population.individuals) < self.population_size:
            parent1 = self.tournament(population)
            parent2 = self.tournament(population)
            child = self.crossover(parent1, parent2)
            new_population.individuals.append(
                self.mutate(child)
            )

        new_population.update_fitness(difficulty)
        new_population.sort_individuals_by_fitness()
        return new_population

    def tournament(self, population: Population) -> List[float]:
        """Selects two random individuals from the population and returns the one with the highest fitness."""
        selected = random.sample(population.individuals, 2)
        if selected[0][-1] > selected[1][-1]:
            return selected[0]
        return selected[1]

    def crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Generates a child by crossing over the weights of two parents."""
        child = [(w1 + w2) / 2 for w1, w2 in zip(parent1[:-1], parent2[:-1])]
        child.append(0.0)  # Initialize fitness value for the child
        return child

    def mutate(self, individual: List[float]) -> List[float]:
        """Mutates or not a random number of positions of the individual according to the mutation rate."""
        if random.random() >= self.mutation_rate:
            # no mutation to this individual
            return individual
        
        # apply mutation
        n_positions_to_mutate = random.randint(1, 5)
        print(f"\tMutating {n_positions_to_mutate} positions...")
        for _ in range(n_positions_to_mutate):
            # -1 to avoid changing the fitness value, and do not change the best value (elitism)
            position = random.randint(1, len(individual) - 1)
            individual[position] = random.uniform(-1.0, 1.0)

        return individual

    def random_difficulty(self) -> Difficulty:
        """Generates a random difficulty according to predefined probabilities."""
        difficulties = [
            (Difficulty.EASY, 0.25),
            (Difficulty.MEDIUM, 0.40),
            (Difficulty.HARD, 0.35)
        ]
        
        choices, weights = zip(*difficulties)
        return random.choices(choices, weights)[0]

if __name__ == "__main__":
    topology = (9, 9, 9)
    ga = GeneticAlgorithm(50, topology, 0.01, 500, True)
    try: ga.run()
    finally:
        print("Saving best individual...")
        nn = NeuralNetwork(topology, ga.population.individuals[0][:-1])
        nn.to_file("model2.json")
import random
from typing import List, Tuple, Optional
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
    def win_condition(row: int, col: int, board: List[List[Player]], player: Player = Player.X) -> bool:
        """
        Checks if the movement made by the player resulted in a win condition.
        That is, if the player has two marks in a row, column, or diagonal and
        the third cell is empty.
        """
        def safe_get(r: int, c: int) -> Optional[Player]:
            """
            Safely get the value from the board at the specified row and column.
            Returns None if the index is out of bounds.
            """
            if 0 <= r < len(board) and 0 <= c < len(board[r]):
                return board[r][c]
            return None

        def check_line(line: List[Optional[Player]]) -> bool:
            return line.count(player) == 2 and line.count(Player.EMPTY) == 1

        # Temporarily place the player's mark
        board[row][col] = player

        # Check row
        if check_line([safe_get(row, i) for i in range(3)]): 
            board[row][col] = Player.EMPTY
            return True

        # Check column
        if check_line([safe_get(i, col) for i in range(3)]): 
            board[row][col] = Player.EMPTY
            return True

        # Check main diagonal
        if row == col and check_line([safe_get(i, i) for i in range(3)]): 
            board[row][col] = Player.EMPTY
            return True

        # Check anti-diagonal
        if row + col == 2 and check_line([safe_get(i, 2 - i) for i in range(3)]): 
            board[row][col] = Player.EMPTY
            return True

        # Remove the temporary mark
        board[row][col] = Player.EMPTY
        return False
    
    def blocked_opponent(row: int, col: int, board: List[List[Player]], player: Player = Player.X) -> bool:
        """
        Checks if the movement made by the player blocked a potential win
        condition of the opponent.
        """
        opponent = Player.O if player == Player.X else Player.X

        def safe_get(row: int, col: int) -> Optional[Player]:
            """
            Safely get the value from the board at the specified row and column.
            Returns None if the index is out of bounds.
            """
            if 0 <= row < len(board) and 0 <= col < len(board[0]):
                return board[row][col]
            return None

        def check_line(line: List[Optional[Player]]) -> bool:
            return line.count(opponent) == 2 and line.count(Player.EMPTY) == 1

        # Check row
        if check_line([safe_get(row, i) for i in range(3)]): 
            return True

        # Check column
        if check_line([safe_get(i, col) for i in range(3)]): 
            return True

        # Check main diagonal
        if row == col and check_line([safe_get(i, i) for i in range(3)]): 
            return True

        # Check anti-diagonal
        if row + col == 2 and check_line([safe_get(i, 2 - i) for i in range(3)]): 
            return True

        return False
    
    fitness: float = 0.0
    board: List[List[Player]] = [[Player.EMPTY for _ in range(3)] for _ in range(3)]
    network: NeuralNetwork = NeuralNetwork(nn_topology, individual)

    # simulate a game between the neural network and the minimax AI
    result: str = classifier.ONGOING
    turns: int = 0
    while result == classifier.ONGOING:
        # 1. neural network plays as X
        row, col = network.predict(_to_float_board(board))

        # 2. evaluate how good the move was
        if board[row][col] != Player.EMPTY:
            # playing at non-empty positions is invalid and results in a game over with a penalty of -90%
            penalty = 0.90 ** turns # exponential decay to value how far the network got
            return fitness * (1 - penalty)

        fitness += 1.0  # valid moves are rewarded by granting a whole fitness point

        '''
        if win_condition(row, col, board):
            # building a strategy towards winning is encouraged by awarding 0.5 fitness points
            fitness += 0.5
        if blocked_opponent(row, col, board):
            # blocking the opponent from winning is also encouraged by awarding 0.5 fitness points
            fitness += 0.5
        if (row, col) == (1, 1):
            # controlling the center is generally a good thing, so that is rewarded with 0.2 fitness points
            fitness += 0.25
        if (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]: 
            # controlling the corners is also generally good, so that is rewarded with 0.1 fitness points
            fitness += 0.2
        '''

        # 3. classify the board after the neural network plays to check for a game over
        board[row][col] = Player.X
        result = classifier.classify(_board_to_string(board))
        if result != classifier.ONGOING: break

        # 4. minimax plays as O
        board = minimax.play(difficulty, board)

        # 5. classify the board after minimax plays to check for a game over at the loop condition
        result = classifier.classify(_board_to_string(board))

        turns += 1

    # game has ended! grant bonus or penalty based on the result of the match
    match result:
        case classifier.X_WON:
            # we award a 50% bonus to the fitness score if the neural network wins
            fitness *= 1.50
        case classifier.TIE:
            # we award a 25% bonus to the fitness score if the match ends in a tie
            fitness *= 1.25
        case classifier.O_WON:
            # we penalize the fitness score with -25% if the neural network loses
            fitness *= 0.75

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
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.eval_fitness, i, difficulty) for i in range(self.size)]
            for future in concurrent.futures.as_completed(futures):
                index, fit_val = future.result()
                self.individuals[index][-1] = fit_val
                print(f"\t{index+1}. Fitness = {fit_val}")

    def eval_fitness(self, index: int, difficulty: Difficulty) -> Tuple[int, float]:
        return index, fitness(
            self.individuals[index][:-1], 
            difficulty, 
            self.topology
        )

    def sort_individuals_by_fitness(self) -> None:
        """Sorts the individuals by their fitness in descending order."""
        self.individuals.sort(key=lambda ind: ind[-1], reverse=True)

    def __len__(self) -> int:
        return len(self.individuals)

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
        self.difficulty = Difficulty.EASY # start with easy difficulty

    def run(self):
        """Runs the genetic algorithm with the parameters specified in the constructor."""
        for g in range(self.max_generations):
            print(f"Training Generation {g+1} with {self.difficulty}...")
            self.population = self.new_generation(self.population, self.difficulty)
            print(f"Generation {g+1} has been trained (best fitness was {self.population.individuals[0][-1]})")
            self.adjust_difficulty()
            print("--------------------------------------------------------------------------------------------")

    def adjust_difficulty(self):
        """Adjusts the difficulty based on the performance of the population."""
        best_fitness = self.population.individuals[0][-1]
        if best_fitness < 1.0:
            self.difficulty = self.random_difficulty(0.70, 0.30, 0.00)
        elif best_fitness < 2.0:
            self.difficulty = self.random_difficulty(0.40, 0.50, 0.10)
        else:
            self.difficulty = self.random_difficulty(0.30, 0.50, 0.20)

    def new_generation(self, population: Population, difficulty: Difficulty) -> Population:
        """Takes in the current population and returns the next generation of individuals."""
        new_population = Population(self.population_size, self.topology)

        if self.elitism:
            population.sort_individuals_by_fitness()
            new_population.individuals.append(population.individuals[0])

        while len(new_population) < self.population_size:
            parent1 = self.tournament(population)
            parent2 = self.tournament(population)
            child = self.crossover(parent1, parent2)
            new_population.individuals.append(child)

        if random.random() < self.mutation_rate:
            self.mutate(new_population)

        new_population.update_fitness(difficulty)
        new_population.sort_individuals_by_fitness()
        return new_population

    def tournament(self, population: Population, tournament_size: int = 10) -> List[float]:
        candidates = random.sample(population.individuals, tournament_size)
        return max(candidates, key=lambda ind: ind[-1])

    def crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Generates a child by crossing over the weights of two parents."""
        child = [(w1 + w2) / 2 for w1, w2 in zip(parent1[:-1], parent2[:-1])]
        child.append(0.0)  # Initialize fitness value for the child
        return child

    def mutate(self, population: Population):
        """
        Mutates a random number of weights of a random number of individuals in the population.
        """
        n_individuals_to_mutate = random.randint(1, 4) # mutate 1 to 4 individuals
        for _ in range(n_individuals_to_mutate):
            individual_idx = random.randint(0, len(population) - 1)
            n_positions_to_mutate = random.randint(1, 4) # mutate 1 to 4 positions
            print(f"\tMutating {n_positions_to_mutate} positions in individual {individual_idx}...")
            for _ in range(n_positions_to_mutate):
                weight_idx = random.randint(0, len(population.individuals[individual_idx]) - 2)
                population.individuals[individual_idx][weight_idx] = random.uniform(-1.0, 1.0)

    def random_difficulty(self, easy_prob: float, medium_prob: float, hard_prob: float) -> Difficulty:
        """Randomizes a difficulty according to the given probabilities."""
        difficulties = [
            (Difficulty.EASY,   easy_prob),
            (Difficulty.MEDIUM, medium_prob),
            (Difficulty.HARD,   hard_prob)
        ]
        choices, weights = zip(*difficulties)
        return random.choices(choices, weights)[0]

if __name__ == "__main__":
    topology = (9, 9, 9)
    population_size = 100
    mutation_rate = 0.10
    max_generations = 500
    elitism = True

    try: 
        ga = GeneticAlgorithm(population_size, topology, mutation_rate, max_generations, elitism)
        ga.run()
    finally:
        print("Saving best individual...")
        nn = NeuralNetwork(topology, ga.population.individuals[0][:-1])
        nn.to_file("model.json")

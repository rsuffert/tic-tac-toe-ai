import random
from typing import List, Tuple
from population import Population
from aiplayers import Difficulty, MinimaxAIPlayer, Player
from tictactoeutils import Classifier

classifier: Classifier = Classifier(
    Player.X.value, Player.O.value, Player.EMPTY.value)

minimax: MinimaxAIPlayer = MinimaxAIPlayer()


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
        self.population.sortIndividualsByFitness()
        best_fitness = self.population.getIndividual(0)[-1]
        print(f"Generation 1: Best fitness = {
            best_fitness}, Difficulty = {difficulty.name}")

        for generation in range(2, self.max_generations + 1):
            difficulty = self.random_difficulty()
            self.population = self.new_generation(self.population, difficulty)
            best_fitness = self.population.getIndividual(0)[-1]
            print(f"Generation {generation}: Best fitness = {
                best_fitness}, Difficulty = {difficulty.name}")

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
        new_population.sortIndividualsByFitness()
        return new_population

    def selection_tournament(self, population: Population) -> List[float]:
        tournament_size = 2
        selected = random.sample(population.individuals, tournament_size)
        selected.sort(key=lambda ind: ind[-1], reverse=True)
        return selected[0]

    def crossover(self, parent1: List[float], parent2: List[float]) -> List[List[float]]:
        child_weights = [(w1 + w2) / 2 for w1,
                         w2 in zip(parent1[:-1], parent2[:-1])]
        child_weights.append(0.0)  # Initialize fitness value for the child
        return [child_weights]

    def mutate(self, individual: List[float]):
        for i in range(len(individual) - 1):
            if random.random() <= self.mutation_rate:
                individual[i] = random.uniform(-1.0, 1.0)

    def random_difficulty(self) -> Difficulty:
        """
        Generates a random difficulty with the following probabilities:
        - 50% chance of being EASY
        - 25% chance of being MEDIUM
        - 25% chance of being HARD
        """
        rand = random.random()
        if rand < 0.3:
            return Difficulty.EASY
        elif rand < 0.66:
            return Difficulty.MEDIUM
        else:
            return Difficulty.HARD


if __name__ == "__main__":
    population_size = 15
    topology = (9, 9, 9)
    crossover_rate = 0.9
    mutation_rate = 0.01
    max_generations = 15
    elitism = True

    ga = GeneticAlgorithm(population_size, topology,
                          crossover_rate, mutation_rate, max_generations, elitism)
    ga.run()

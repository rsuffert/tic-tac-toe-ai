import random
from typing import List, Tuple
from population import Population
from neural_network import NeuralNetwork
from aiplayers import Difficulty


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
        generation = 1
        while generation <= self.max_generations:
            self.population.updateFitness(Difficulty.EASY)
            self.population.sortIndividualsByFitness()
            best_fitness = self.population.getIndividual(0)[-1]
            print(f"Generation {generation}: Best fitness = {best_fitness}")
            self.population = self.new_generation(self.population)
            generation += 1

    def new_generation(self, population: Population) -> Population:
        new_population = Population(self.population_size, self.topology)
        if self.elitism:
            new_population.setIndividual(0, population.getIndividual(0))

        while len(new_population.individuals) < self.population_size:
            parents = self.selection_tournament(population)
            if random.random() <= self.crossover_rate:
                offspring = self.crossover(parents[0], parents[1])
            else:
                offspring = parents

            for child in offspring:
                self.mutate(child)
                new_population.individuals.append(child)

        new_population.updateFitness(Difficulty.EASY)
        new_population.sortIndividualsByFitness()
        return new_population

    def selection_tournament(self, population: Population) -> List[List[float]]:
        tournament_size = 2
        selected = random.sample(population.individuals, tournament_size)
        selected.sort(key=lambda ind: ind[-1], reverse=True)
        return selected[:2]

    def crossover(self, parent1: List[float], parent2: List[float]) -> List[List[float]]:
        crossover_point = random.randint(0, len(parent1) - 1)
        child1_weights = parent1[:crossover_point] + \
            parent2[crossover_point:-1] + [0.0]
        child2_weights = parent2[:crossover_point] + \
            parent1[crossover_point:-1] + [0.0]
        return [child1_weights, child2_weights]

    def mutate(self, individual: List[float]):
        for i in range(len(individual) - 1):
            if random.random() <= self.mutation_rate:
                individual[i] = random.uniform(-1.0, 1.0)


if __name__ == "__main__":
    population_size = 100
    topology = (9, 9, 9)
    crossover_rate = 0.7
    mutation_rate = 0.01
    max_generations = 1000
    elitism = True

    ga = GeneticAlgorithm(population_size, topology,
                          crossover_rate, mutation_rate, max_generations, elitism)
    ga.run()

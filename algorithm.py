import random
import pandas as pd
import numpy as np
from math import ceil, log2, pow
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GeneticAlgorithm:
    def __init__(self, initial_value, final_value, generations, optimization, individual_mutation, mutation_per_gen,
                 initial_limit_x, maximum_limit_x, reference_resolution, ):
        self.initial_value = initial_value
        self.final_value = final_value
        self.generations = generations
        self.optimization = optimization
        self.individual_mutation = individual_mutation
        self.mutation_per_gen = mutation_per_gen
        self.initial_limit_x = initial_limit_x
        self.maximum_limit_x = maximum_limit_x
        self.reference_resolution = reference_resolution
  

        self.best_global_individual = None
        self.last_population = None

        if self.initial_limit_x >= self.maximum_limit_x:
            raise ValueError("initial_limit_x debe ser menor que maximum_limit_x")

        self.range = self.maximum_limit_x - self.initial_limit_x
        self.bits = ceil(log2(self.range / self.reference_resolution + 1))
        self.system_resolution = self.range / (pow(2, self.bits) - 1)

   
        self.data = pd.read_excel('data.xlsx' , skiprows=1)
        self.data.columns = self.data.columns.str.strip()
        self.x1 = self.data['x1'].values
        self.x2 = self.data['x2'].values
        self.x3 = self.data['x3'].values
        self.x4 = self.data['x4'].values
        self.Y = self.data['y'].values

    def generate_population(self):
        population = []
        while len(population) < self.initial_value:
            individual = [random.uniform(self.initial_limit_x, self.maximum_limit_x) for _ in range(5)]
            population.append(individual)
        return population

    def fitness(self, individual):
        A, B, C, D, E = individual
        Y_pred = A + B * self.x1 + C * self.x2 + D * self.x3 + E * self.x4
        return np.mean((self.Y - Y_pred) ** 2)

    def select_pairs(self, population):
        fitness_values = [(self.fitness(individual), individual) for individual in population]
        fitness_values.sort(key=lambda x: x[0])
        # Asegurarse de que self.initial_value es un número entero
        initial_value = int(self.initial_value)
        selected = [individual for _, individual in fitness_values[:initial_value // 2]]
        pairs = []
        for individual in selected:
            partner = random.choice(selected)
            while partner == individual:
                partner = random.choice(selected)
            pairs.append((individual, partner))
        return pairs

    def crossover(self, pairs):
        children = []
        for pair in pairs:
            parent1, parent2 = pair
            crossover_point = random.randint(1, 4)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            children.append(child1)
            children.append(child2)
        return children

    def mutation(self, individual):
        mutated_individual = individual.copy()
        for i in range(len(mutated_individual)):
            if random.random() < self.mutation_per_gen:
                mutated_individual[i] = random.uniform(self.initial_limit_x, self.maximum_limit_x)
        return mutated_individual

    def population_mutation(self, population):
        mutated_population = []
        for individual in population:
            if random.random() < self.individual_mutation:
                mutated_population.append(self.mutation(individual))
            else:
                mutated_population.append(individual)
        return mutated_population

    def find_best_individual(self, population):
        best_individual = min(population, key=self.fitness)
        return best_individual

    def prune_population(self, population):
        if self.best_global_individual not in population:
            population.append(self.best_global_individual)
        population.sort(key=self.fitness)
        # Asegurarse de que self.final_value es un número entero
        final_value = int(self.final_value)
        return population[:final_value]

    def run(self):
        population = self.generate_population()

        generation_counter = 1
        all_generations_info = []

        for _ in range(self.generations):
            pairs_population = self.select_pairs(population)
            crossover_population = self.crossover(pairs_population)
            mutation_population = self.population_mutation(crossover_population)
            population_generate = population + mutation_population
            best_individual = self.find_best_individual(population_generate)

            if self.best_global_individual is None or self.fitness(best_individual) < self.fitness(self.best_global_individual):
                self.best_global_individual = best_individual
                print(f"Generación {generation_counter}: El mejor individuo global se ha actualizado a {self.best_global_individual}")

            fitness_generation = [self.fitness(ind) for ind in population_generate]
            best_fitness = min(fitness_generation)
            worst_fitness = max(fitness_generation)
            average_fitness = sum(fitness_generation) / len(fitness_generation)

            generation_info = {
                'generation': generation_counter,
                'population': population_generate,
                'pairs_population': pairs_population,
                'crossover_population': crossover_population,
                'mutation_population': mutation_population,
                'best_individual': best_individual,
                'best_global_individual': self.best_global_individual,
                'best_fitness': best_fitness,
                'worst_fitness': worst_fitness,
                'average_fitness': average_fitness
            }

            all_generations_info.append(generation_info)
            population = self.prune_population(population_generate)

            generation_counter += 1
        A, B, C, D, E = self.best_global_individual
        print(f"Las constantes del mejor individuo son: A={A}, B={B}, C={C}, D={D}, E={E}")


      

        return all_generations_info

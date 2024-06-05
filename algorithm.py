import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, initial_value, final_value, generations, individual_mutation, mutation_per_gen):
        self.initial_value = int(initial_value)
        self.final_value = int(final_value)
        self.generations = generations
        self.individual_mutation = individual_mutation
        self.mutation_per_gen = mutation_per_gen

        self.best_global_individual = None
        self.last_population = None
        # Leer el archivo Excel
        self.data = pd.read_excel('data1.xlsx')
        self.data.columns = ['x1', 'x2', 'x3', 'x4', 'y']

        self.x1 = self.data['x1'].values
        self.x2 = self.data['x2'].values
        self.x3 = self.data['x3'].values
        self.x4 = self.data['x4'].values
        self.Y = self.data['y'].values

    def reset(self):
        self.best_global_individual = None
        self.last_population = None

    def generate_population(self):
        population = [[random.uniform(-10, 10) for _ in range(5)] for _ in range(self.initial_value)]
        return population

    def fitness(self, individual):
        A, B, C, D, E = individual
        Y_pred = A + B * self.x1 + C * self.x2 + D * self.x3 + E * self.x4
        return np.mean(np.abs(self.Y - Y_pred))

    def select_pairs(self, population):
        fitness_values = [(self.fitness(individual), individual) for individual in population]
        fitness_values.sort(key=lambda x: x[0])
        selected = [individual for _, individual in fitness_values]
        pairs = []
        for individual in selected:
            possible_partners = [ind for ind in selected if ind != individual]
            if possible_partners:
                partner = random.choice(possible_partners)
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
                u = random.uniform(-1, 1)
                # Limit the mutation based on the fitness error
                error = self.fitness(individual)
                mutation_range = -50 if error > 1 else -error * 50  # Adjust this as needed
                mutated_individual[i] = mutated_individual[i] * (1.0 + u * random.uniform(mutation_range, 50)/100)
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
        return min(population, key=self.fitness)

    def prune_population(self, population):
        population.sort(key=self.fitness)
        return population[:self.final_value]

    def run(self):
        self.reset()
        population = self.generate_population()
        results = []
        errors = []

        for gen in range(self.generations):
            pairs_population = self.select_pairs(population)
            crossover_population = self.crossover(pairs_population)
            mutation_population = self.population_mutation(crossover_population)
            population_generate = population + mutation_population
            best_individual = self.find_best_individual(population_generate)
            best_individual_error = self.fitness(best_individual)

            if self.best_global_individual is None or best_individual_error < self.fitness(self.best_global_individual):
                self.best_global_individual = best_individual

            errors.append(best_individual_error)

            self.last_population = self.prune_population(population_generate)
            if self.best_global_individual not in self.last_population:
                self.last_population.append(self.best_global_individual)
            population = self.last_population

            results.append({
                'generation': gen + 1,
                'best_individual': f"A={best_individual[0]}, B={best_individual[1]}, C={best_individual[2]}, D={best_individual[3]}, E={best_individual[4]}",
                'best_individual_error': best_individual_error
            })

        best_individual_error = self.fitness(self.best_global_individual)
        results.append({
            'generation': 'global',
            'best_individual': f"A={self.best_global_individual[0]}, B={self.best_global_individual[1]}, C={self.best_global_individual[2]}, D={self.best_global_individual[3]}, E={self.best_global_individual[4]}",
            'best_individual_error': best_individual_error
        })

        # Plot the errors
        plt.figure()
        plt.plot(errors)
        plt.xlabel('Generación')
        plt.ylabel('Error absoluto')
        plt.title('Evolución del error absoluto')
        plt.show()

        return results, self.last_population

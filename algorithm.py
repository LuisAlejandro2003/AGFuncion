import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

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
        self.data = pd.read_excel('Dataset04.xlsx')
        self.data.columns = ['ID', 'x1', 'x2', 'x3', 'x4', 'x5', 'y']

        self.x1 = self.data['x1'].values
        self.x2 = self.data['x2'].values
        self.x3 = self.data['x3'].values
        self.x4 = self.data['x4'].values
        self.x5 = self.data['x5'].values
        self.Y = self.data['y'].values

    def reset(self):
        self.best_global_individual = None
        self.last_population = None

    def generate_population(self):
        population = [[random.uniform(-10, 10) for _ in range(6)] for _ in range(self.initial_value)]
        return population

    # Error Absoluto Medio (Promedio de las diferencias absolutas en ambas Ys)
    def fitness(self, individual):
        A, B, C, D, E, F = individual
        Y_pred = A + B * self.x1 + C * self.x2 + D * self.x3 + E * self.x4 + F * self.x5
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
            crossover_point = random.randint(1, 5)
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
                error = self.fitness(individual)
                mutation_range = -50 if error > 1 else -error * 50
                mutated_individual[i] = mutated_individual[i] * (1.0 +  random.uniform(mutation_range, 50)/100)
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

    def plot_comparison(self, Y_real, Y_pred_values):
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.plot(Y_real, color='red', label='Valores Reales')
            ax.plot(Y_pred_values[i], color='blue', label='Predicción Generación')
            ax.set_xlabel('ID')
            ax.set_ylabel('Valores de Y')
            ax.set_title(f'Comparación de Valores Reales y Predichos - Generación {i+1}')
            ax.legend()

        ani = animation.FuncAnimation(fig, animate, frames=len(Y_pred_values), repeat=False)
        ani.save('comparison.mp4', writer='ffmpeg')

    def run(self):
        self.reset()
        population = self.generate_population()
        results = []
        errors = []
        best_errors = []
        worst_errors = []
        avg_errors = []

        # Almacenar los coeficientes del mejor individuo en cada generación
        A_values = []
        B_values = []
        C_values = []
        D_values = []
        E_values = []
        F_values = []

        # Almacenar los valores predichos de Y para el mejor individuo en cada generación
        Y_pred_values = []

        for gen in range(self.generations):
            pairs_population = self.select_pairs(population)
            crossover_population = self.crossover(pairs_population)
            mutation_population = self.population_mutation(crossover_population)
            population_generate = population + mutation_population
            best_individual = self.find_best_individual(population_generate)

            # Calcular promedios, mejor y peor
            best_individual_error = self.fitness(best_individual)
            worst_individual_error = self.fitness(max(population_generate, key=self.fitness))
            avg_individual_error = np.mean([self.fitness(ind) for ind in population_generate])
            # agregarlos
            best_errors.append(best_individual_error)
            worst_errors.append(worst_individual_error)
            avg_errors.append(avg_individual_error)

            # Si el mejor individuo global no está en la población actual lo agrega
            if self.best_global_individual is None or best_individual_error < self.fitness(self.best_global_individual):
                self.best_global_individual = best_individual

            errors.append(best_individual_error)

            # Almacenar los coeficientes del mejor individuo
            A_values.append(best_individual[0])
            B_values.append(best_individual[1])
            C_values.append(best_individual[2])
            D_values.append(best_individual[3])
            E_values.append(best_individual[4])
            F_values.append(best_individual[5])

            # Calcular y almacenar los valores predichos de Y para el mejor individuo
            A, B, C, D, E, F = best_individual
            Y_pred = A + B * self.x1 + C * self.x2 + D * self.x3 + E * self.x4 + F * self.x5
            Y_pred_values.append(Y_pred)

            # PODA
            self.last_population = self.prune_population(population_generate)
            if self.best_global_individual not in self.last_population:
                self.last_population.append(self.best_global_individual)
            population = self.last_population
            # regresa todos los resultados
            results.append({
                'generation': gen + 1,
                'best_individual': f"A={best_individual[0]}, B={best_individual[1]}, C={best_individual[2]}, D={best_individual[3]}, E={best_individual[4]}, F={best_individual[5]}",
                'best_individual_error': best_individual_error
            })

        # Llama a plot_comparison para la gráfica después del bucle de generaciones
        self.plot_comparison(self.Y, Y_pred_values)

        # Plot the errors
        plt.figure()
        plt.plot(best_errors, label='Mejor error')
        plt.plot(worst_errors, label='Peor error')
        plt.plot(avg_errors, label='Error promedio')
        plt.xlabel('Generación')
        plt.ylabel('Error absoluto')
        plt.title('Evolución del error absoluto')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(A_values, label='A')
        plt.plot(B_values, label='B')
        plt.plot(C_values, label='C')
        plt.plot(D_values, label='D')
        plt.plot(E_values, label='E')
        plt.plot(F_values, label='F')
        plt.xlabel('Generación')
        plt.ylabel('Valor del coeficiente')
        plt.title('Evolución de los coeficientes del mejor individuo')
        plt.legend()
        plt.show()

        return results, self.last_population
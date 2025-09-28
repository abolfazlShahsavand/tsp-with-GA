import random
from typing import List, Callable, Tuple


class GeneticAlgorithm:
    def __init__(
        self,
        population: List[List[int]],
        fitness_func: Callable[[List[int]], float],
        num_generations: int = 100,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        selection_strategy: str = 'tournament',  # 'tournament', 'roulette', 'rank'
        crossover_strategy: str = 'uniform',  # 'one_point', 'two_point', 'uniform'
        mutation_strategy: str = 'swap',        # 'swap', 'inversion', 'scramble'
        tournament_size: int = 3,
) -> None:
        """
        Initialize the genetic algorithm with population and configuration parameters.

        Args:
            population (List[List[int]]): Initial population of individuals.
            fitness_func (Callable[[List[int]], float]): Function to evaluate the fitness of an individual.
            num_generations (int): Number of generations to run the algorithm.
            mutation_rate (float): Probability of mutation for each individual.
            crossover_rate (float): Probability of crossover between pairs.
            selection_strategy (str): Selection method ('tournament', 'roulette', or 'rank').
            crossover_strategy (str): Crossover method ('one_point', 'two_point', or 'uniform').
            mutation_strategy (str): Mutation method ('swap', 'inversion', or 'scramble').
            tournament_size (int): Size of the tournament in tournament selection.
        """
        self.population: List[List[int]] = population
        self.fitness_func: Callable[[List[int]], float] = fitness_func
        self.num_generations: int = num_generations
        self.mutation_rate: float = mutation_rate
        self.crossover_rate: float = crossover_rate
        self.selection_strategy: str = selection_strategy
        self.crossover_strategy: str = crossover_strategy
        self.mutation_strategy: str = mutation_strategy
        self.tournament_size: int = tournament_size

        # History tracking
        self.best_fitness_history: List[float] = []
        self.best_distance_history: List[float] = []

    def evaluate_population(self) -> List[float]:
        """
        Evaluate the fitness of the current population.

        Returns:
            List[float]: A list containing the fitness value of each individual.
        """
        return [self.fitness_func(individual) for individual in self.population]
    pass
        

    def selection(self, fitnesses: List[float]) -> int:
        """
        Select an individual index based on the configured selection strategy.

        Args:
            fitnesses (List[float]): List of fitness values for the population.

        Returns:
            int: Index of the selected individual.
        """
        if self.selection_strategy == 'tournament':
            participants = random.sample(range(len(fitnesses)), self.tournament_size)
            best = max(participants, key=lambda idx: fitnesses[idx])
            return best
        
        elif self.selection_strategy == 'roulette':
            total_fitness = sum(fitnesses)
            pick = random.uniform(0, total_fitness)
            current = 0
            for i, f in enumerate(fitnesses):
                current += f
                if current >= pick:
                    return i

        elif self.selection_strategy == 'rank':
            sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
            ranks = list(range(1, len(fitnesses) + 1))
            total_rank = sum(ranks)
            probs = [r / total_rank for r in ranks]
            pick = random.random()
            current = 0
            for i, idx in enumerate(sorted_indices):
                current += probs[i]
                if current >= pick:
                    return idx
        
        pass

        

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Perform permutation-preserving crossover between two parents.

        Args:
            parent1 (List[int]): First parent individual.
            parent2 (List[int]): Second parent individual.

        Returns:
            List[int]: Offspring individual produced by crossover.
        """
        size = len(parent1)
        if self.crossover_strategy == 'one_point':
            point = random.randint(1, size - 1)
            child = parent1[:point]
            child += [gene for gene in parent2 if gene not in child]
            return child

        elif self.crossover_strategy == 'two_point':
            i, j = sorted(random.sample(range(size), 2))
            child = [-1] * size
            child[i:j+1] = parent1[i:j+1]
            pos = 0
            for gene in parent2:
                if gene not in child:
                    while child[pos] != -1:
                        pos += 1
                    child[pos] = gene
            return child

        elif self.crossover_strategy == 'uniform':
            child = [-1] * size
            taken = set()
            for i in range(size):
                gene = parent1[i] if random.random() < 0.5 else parent2[i]
                if gene not in taken:
                    child[i] = gene
                    taken.add(gene)
            for i in range(size):
                if child[i] == -1:
                    for gene in parent1 + parent2:
                        if gene not in taken:
                            child[i] = gene
                            taken.add(gene)
                            break
            return child    

    def mutation(self, individual: List[int]) -> List[int]:
        """
        Apply mutation to an individual using the configured mutation strategy.

        Args:
            individual (List[int]): Individual to mutate.

        Returns:
            List[int]: Mutated individual.
        """
        mutated = individual.copy()
        
        if self.mutation_strategy == 'swap':
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]

        elif self.mutation_strategy == 'inversion':
            i, j = sorted(random.sample(range(len(mutated)), 2))
            mutated[i:j+1] = list(reversed(mutated[i:j+1]))

        elif self.mutation_strategy == 'scramble':
            i, j = sorted(random.sample(range(len(mutated)), 2))
            subset = mutated[i:j+1]
            random.shuffle(subset)
            mutated[i:j+1] = subset

        return mutated
        
    def run(self) -> Tuple[List[int], float, List[float], List[float]]:
        """
        Run the genetic algorithm for the configured number of generations.

        Returns:
            Tuple containing:
                - List[int]: The best solution found.
                - float: Fitness of the best solution.
                - List[float]: History of best fitness values.
                - List[float]: History of best distance values.
        """
        best_individual = None
        best_fitness = float('-inf')

        for generation in range(self.num_generations):
            fitnesses = self.evaluate_population()

            max_fitness = max(fitnesses)
            best_idx = fitnesses.index(max_fitness)
            current_best = self.population[best_idx]

            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_individual = current_best

            self.best_fitness_history.append(max_fitness)
            self.best_distance_history.append(1 / max_fitness if max_fitness > 0 else float('inf'))

            new_population = []

            while len(new_population) < len(self.population):
                idx1 = self.selection(fitnesses)
                idx2 = self.selection(fitnesses)
                parent1, parent2 = self.population[idx1], self.population[idx2]

                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                if random.random() < self.mutation_rate:
                    child = self.mutation(child)

                new_population.append(child)

            self.population = new_population

        return best_individual, best_fitness, self.best_fitness_history, self.best_distance_history

        

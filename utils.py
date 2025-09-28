
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
import math

def read_dataset(filename: str) -> List[Tuple[float, float]]:
    with open("data/" + filename + ".tsp", 'r') as f:
        cities: List[Tuple[float, float]] = []
        for line in f:
            parts = line.split()
            if len(parts) == 3 and ":" not in parts:
                cities.append((float(parts[1]), float(parts[2])))
    return cities
    pass


def create_distance_matrix(cities: List[Tuple[float, float]]) -> List[List[float]]:

    n = len(cities)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = cities[i]
                x2, y2 = cities[j]
                dist = math.hypot(x2 - x1, y2 - y1)
                matrix[i][j] = dist
            else:
                matrix[i][i] = 0.0
    return matrix
    pass


def calculate_route_distance(route: List[int], distance_matrix: List[List[float]]) -> float:

    total_distance = 0.0
    num_cities = len(route)

    for i in range(num_cities):
        from_city = route[i]
        to_city = route[(i + 1) % num_cities]  
        total_distance += distance_matrix[from_city][to_city]

    return total_distance
    pass

def fitness(route: List[int], distance_matrix: List[List[float]]) -> float:
    """
    Defines fitness as the inverse of the total route distance.

    Args:
        route (List[int]): A permutation representing a tour.
        distance_matrix (List[List[float]]): Precomputed pairwise distance matrix.

    Returns:
        float: Fitness value, higher is better.
    """
    total_distance = calculate_route_distance(route, distance_matrix)
    if total_distance == 0:
        return float('inf')
    return 1 / total_distance
    pass


def generate_initial_population(num_individuals: int, num_cities: int) -> List[List[int]]:
    """
    Generates `num_individuals` random tours over `num_cities` cities.

    Args:
        num_individuals (int): Number of individuals in the population.
        num_cities (int): Number of cities in each tour.

    Returns:
        List[List[int]]: Initial population, a list of random permutations.
    """
    population = []
    base_route = list(range(num_cities))

    for _ in range(num_individuals):
        route = base_route.copy()
        random.shuffle(route) 
        population.append(route)

    return population


def plot_route(cities, route=None):
    plt.figure(figsize=(10, 6))
    
    if route is not None:
        for i in range(len(route)):
            start = cities[route[i]]
            end   = cities[route[(i + 1) % len(route)]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'r-')

        # Highlight start and end nodes
        start_node = cities[route[0]]
        end_node   = cities[route[-1]]
        plt.scatter(*start_node, c='blue', s=150, marker='*', label='Start')
        # plt.text(start_node[0] + 0.1, start_node[1] + 0.02, "Start", fontsize=9, color='green')
        
        plt.scatter(*end_node, c='red', s=150, marker='X', label='End')
        # plt.text(end_node[0] + 0.1, end_node[1] + 0.02, "End", fontsize=9, color='red')

    # Plot all cities and their indices
    xs = [pt[0] for pt in cities]
    ys = [pt[1] for pt in cities]
    plt.scatter(xs, ys, c='blue', marker='o')
    for i, (x, y) in enumerate(cities):
        plt.text(x + 0.02, y + 0.02, str(i), fontsize=9, color='black')

    plt.title("2D Map of Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.show()

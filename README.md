# Traveling Salesman Problem Solver using Genetic Algorithm

## Overview
This project implements a Genetic Algorithm to solve the Traveling Salesman Problem (TSP), finding the shortest route that visits each city exactly once and returns to the origin city.

## Project Structure
- `genetic.py` - Core Genetic Algorithm implementation
- `main.py` - Main execution script
- `utils.py` - Utility functions (distance calculation, visualization)
- `data/` - City coordinate datasets
- `.idea/` - IDE configuration files

## Installation
```bash
git clone https://github.com/abolfazlShahsavand/tsp-with-GA.git
cd tsp-with-GA
pip install numpy matplotlib
```
## Usage

Run the solver:
bash

python main.py

Or use as a module:
python

from genetic import GeneticAlgorithm
from utils import load_cities

cities = load_cities('data/cities.txt')
ga = GeneticAlgorithm(cities)
best_route, best_distance = ga.evolve()

## Features

    Multiple selection methods (Tournament, Roulette Wheel)

    Various crossover operators (Order Crossover, PMX)

    Mutation operations (Swap, Inversion)

    Customizable parameters (population size, mutation rate, generations)

    Route visualization and performance metrics

## Configuration

Adjustable parameters include:

    Population size: 100

    Generations: 500

    Mutation rate: 0.02

    Crossover rate: 0.8

    Tournament size: 5

    Elitism count: 2

# Results

The algorithm provides:

    Optimal route visualization

    Convergence graphs

    Performance statistics

    Distance metrics

## Author

## Abolfazl Shahsavand

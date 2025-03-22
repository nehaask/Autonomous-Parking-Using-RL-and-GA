# Genetic Algorithm for Optimal Trajectory Planning

This repository implements a genetic algorithm (GA) to optimize trajectory planning using control parameters (gamma and beta). The algorithm evolves these parameters over multiple generations to find an optimal control strategy for a given dynamic system.

## Features
- **Binary Encoding & Gray Code Conversion**: Parameters are encoded in binary and converted to Gray code for efficient crossover.
- **Fitness Evaluation via an ODE System**: The fitness function is computed by simulating trajectories using Euler's method.
- **Genetic Operators**: Selection, crossover, and mutation are applied to evolve the control parameters.
- **Visualization**: Generates trajectory plots and evolution graphs for control parameters.

## Installation

## Usage
Run the main script to execute the genetic algorithm:
```sh
python source_code.py
```
This will simulate the evolutionary process and output the optimized trajectory.

## Configuration
You can modify key parameters in the script:
- `POP_SIZE`: Population size
- `MAX_GEN`: Maximum number of generations
- `MUTATION_RATE`: Mutation probability
- `MAX_TIME`: Time constraint for execution

## Output
- **Trajectory Plots**: Shows the evolution of the optimal trajectory.
- **Parameter Evolution Graphs**: Visualizes how gamma and beta change over generations.

## Contributing
Feel free to submit issues or pull requests to improve this project!

## License
This project is licensed under the MIT License.


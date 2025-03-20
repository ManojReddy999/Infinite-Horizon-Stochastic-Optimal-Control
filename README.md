# ECE 276B Project 3: Infinite-Horizon Stochastic Optimal Control

This repository contains the code for Project 3 of ECE 276B: Planning & Learning in Robotics (Winter 2024), focusing on infinite-horizon stochastic optimal control for a differential-drive robot. The project implements and compares two control strategies: Certainty Equivalent Control (CEC) and Generalized Policy Iteration (GPI).

**Author:** Manoj Kumar Reddy Manchala (mmanchala@ucsd.edu)

## Project Overview

The project's objective is to design a control policy for a differential-drive robot to track a predefined Lissajous trajectory while avoiding two circular obstacles.  The robot's dynamics are modeled as a discrete-time stochastic system with Gaussian noise.

The project explores two main approaches:

1.  **Receding-Horizon Certainty Equivalent Control (CEC):**  A suboptimal control strategy that simplifies the stochastic problem by assuming deterministic dynamics (ignoring noise) at each time step.  It solves a finite-horizon deterministic optimal control problem using nonlinear programming (NLP) at each step, replanning as the robot moves.  This leverages CasADi for optimization.

2.  **Generalized Policy Iteration (GPI):**  A dynamic programming approach that finds an optimal policy by iteratively evaluating and improving a value function.  The state and control spaces are discretized, and transition probabilities and stage costs are precomputed.  Parallel computing (using Ray) is employed to manage the computational complexity.  *Note: The provided GPI implementation is incomplete and serves as a starting point, not a fully working solution.*

## Files Included

*   **`main.py`:**  The main script that runs the simulation.  It initializes the robot, sets up the control loop, and calls the CEC controller (or, potentially, a GPI controller). Includes visualization setup.
*   **`cec.py`:**  Implements the receding-horizon CEC controller.  Uses CasADi to formulate and solve the NLP problem.
*   **`gpi.py`:**  *Partially implemented* GPI algorithm. Includes classes for configuration (`GpiConfig`) and the GPI controller itself.  Includes (commented-out) code for precomputing transition matrices and stage costs using Ray, as well as placeholder functions for policy evaluation and improvement.
*   **`value_function.py`:** Defines abstract and concrete `ValueFunction` classes, including `GridValueFunction` (for the grid-based GPI) and `FeatureValueFunction` (for a potential feature-based approach, as discussed in the project description).
*   **`utils.py`:**  Provides utility functions, including:
    *   `lissajous(k)`:  Generates the reference trajectory (a Lissajous curve).
    *   `simple_controller(cur_state, ref_state)`:  A basic proportional controller (used as a placeholder or for comparison).
    *   `car_next_state(time_step, cur_state, control, noise=True)`:  Simulates the robot's dynamics (discrete-time, with noise).
    *   `car_next_state_casadi(time_step, cur_state, control)`: CasADi version of the dynamics, for use in the CEC controller.
    *   `visualize(car_states, ref_traj, obstacles, t, time_step, save=False)`:  Creates a visualization of the robot's trajectory, reference trajectory, and obstacles.  Uses Matplotlib.
    *   `timer(func)`: A decorator for timing function execution.
* **`Transition_Matrix.py`**: Precomputes and stores the transition matrix for GPI.
* **`Stage_Costs.py`**: Precomputes and stores the stage costs for GPI.
* **`transition_matrix_f16.npy`**: Precomputed transition matrix.
* **`stage_cost_f16.npy`**: Precomputed stage costs.
* **`ECE 276B_PR3_Report.pdf`**: Project Report

## Dependencies

*   **Python 3.x**
*   **NumPy:**  For numerical computation.
*   **Matplotlib:**  For visualization.
*   **CasADi:**  For nonlinear programming (used in `cec.py`).  Install with `pip install casadi`.
*   **Ray:**  For parallel computing (used in `gpi.py`).  Install with `pip install ray`.
*   **tqdm:**  For progress bars (optional, used in `utils.py`). Install with `pip install tqdm`

## Running the Code

1.  **Install Dependencies:**  Make sure you have all the required libraries installed.

    ```bash
    pip install numpy matplotlib casadi ray tqdm
    ```

2.  **Run `main.py`:**

    ```bash
    python main.py
    ```

    This will:
    *   Initialize the robot's state.
    *   Run the simulation loop, calling the `cec.py` controller at each step.
    *   Print out timing information and final errors.
    *   Generate a visualization of the robot's trajectory.  By default, it will display the plot;  you can set `save=True` in the `utils.visualize` call to save it as a GIF.

**Important Notes:**

*   **GPI Implementation:** The provided `gpi.py` code is *incomplete*. The core GPI logic (policy evaluation and improvement) is not fully implemented, particularly the parallelized updates. The transition matrix and stage cost computation functions are present (and parallelized), but the policy evaluation is a placeholder, and policy improvement has errors.
*   **CEC Tuning:** The performance of the CEC controller depends heavily on the parameters (horizon length `T`, discount factor `gamma`, cost matrices `Q` and `R`, and the scalar `q`). You will need to experiment with these parameters to achieve good trajectory tracking and obstacle avoidance.
*   **CasADi:** The CEC controller uses CasADi for optimization. Understanding CasADi's syntax is essential for modifying or debugging the `cec.py` code.
*   **Computational Cost:** The GPI algorithm, *when fully implemented*, can be computationally expensive, especially with a fine discretization of the state and control spaces. The provided code includes attempts to mitigate this using precomputation and parallelization (with Ray), but it may still take a considerable amount of time to run. The CEC approach is significantly faster.
*   **Precomputed Data:** The provided transition matrix and stage costs are stored as `.npy` files. These are loaded by `gpi.py`. If you change the discretization parameters, you will need to regenerate these files, by running `Transition_Matrix.py` and `Stage_Costs.py`
* **Report:** Read `ECE 276B_PR3_Report.pdf` for the complete report.

This README provides a comprehensive guide to the code and how to run it. Remember to consult the project description for the full theoretical background and requirements.
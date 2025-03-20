import os
from utils import lissajous
import ray
import numpy as np  

ex_space=np.linspace(-3, 3, 21),  # Example discretization, change as needed
ey_space=np.linspace(-3, 3, 21),
eth_space=np.linspace(-np.pi, np.pi, 40),
v_space=np.linspace(0, 1, 6),  # Example control space, change as needed
w_space=np.linspace(-1, 1, 11)

Q = np.eye(2)
q = 1.0
R = np.eye(2)

def compute_stage_cost(error_state, control):
    p_error = error_state[:2]
    theta_error = error_state[2]
    return p_error.T @ Q @ p_error + q * (1 - np.cos(theta_error)) ** 2 + control.T @ R @ control

def compute_stage_costs():
    """
    Compute the stage costs in advance to speed up the GPI algorithm.
    """
    nt, nx, ny, ntheta = 100, 21, 21, 40
    nv, nw = 6, 11

    @ray.remote
    def worker(start, end):
        local_costs = np.empty((end-start, nx, ny, ntheta, nv, nw))
        for t in range(end-start):
            for ex in range(nx):
                for ey in range(ny):
                    for eth in range(ntheta):
                        for v in range(nv):
                            for w in range(nw):
                                error_state = np.array([ex_space[0][ex], ey_space[0][ey], eth_space[0][eth]])
                                control = np.array([v_space[0][v],w_space[w]])
                                local_costs[t, ex, ey, eth, v, w] = compute_stage_cost(error_state, control)
        return local_costs

    num_workers = 32
    jobs = []
    for i in range(num_workers):
        start = i * (nt // num_workers)
        end = (i + 1) * (nt // num_workers)
        jobs.append(worker.remote(start, end))

    results = ray.get(jobs)
    ray.shutdown()

    stage_costs = np.empty((nt, nx, ny, ntheta, nv, nw), dtype=np.float16)
    for j in range(num_workers):
        start = j * (nt // num_workers)
        end = (j + 1) * (nt // num_workers)
        stage_costs[start:end] = results[j]

    return stage_costs

stage_cost_file = '/home/mmkr/Documents/ECE276B/ECE276B_PR3/starter_code/stage_cost_f16.npy'  # Specify the file path to save the array

if not os.path.exists(stage_cost_file):
    with open(stage_cost_file, 'w') as file:
        pass

stage_cost = compute_stage_costs()
np.save(stage_cost_file, stage_cost)
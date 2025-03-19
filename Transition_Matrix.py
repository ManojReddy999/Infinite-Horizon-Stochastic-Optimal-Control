import os
from utils import lissajous
import ray
import numpy as np  

ex_space=np.linspace(-3, 3, 21),  # Example discretization, change as needed
ey_space=np.linspace(-3, 3, 21),
eth_space=np.linspace(-np.pi, np.pi, 40),
v_space=np.linspace(0, 1, 6),  # Example control space, change as needed
w_space=np.linspace(-1, 1, 11)


def compute_neighbors_and_probabilities(next_state_mean):
    std = np.array([0.04, 0.04, 0.004])
    offsets = np.array([[-1, -1, 0], [-1, 0, 0], [-1, 1, 0],
                        [0, -1, 0], [0, 1, 0],
                        [1, -1, 0], [1, 0, 0], [1, 1, 0]])
    neighbors = next_state_mean + offsets

    probabilities = gaussian_likelihood(neighbors, next_state_mean, std)
    probabilities /= probabilities.sum()

    return np.hstack((neighbors, probabilities[:, np.newaxis]))

def gaussian_likelihood(x, mean, std):
    # Handle multiple rows in x
    return np.exp(-0.5 * np.sum(((x - mean) / std) ** 2, axis=1)) / (np.sqrt(2 * np.pi) * np.prod(std))


def compute_next_state_mean(e, ref_cur,ref_next, u, dt=0.1):
    # Create the g matrix
    g = np.array([
        [dt * np.cos(e[2] + ref_cur[2]), 0],
        [dt * np.sin(e[2] + ref_cur[2]), 0],
        [0, dt]
    ])
    
    # Calculate the new state
    return e + np.dot(g, u.T) + np.array(ref_cur) - np.array(ref_next)


def compute_transition_matrix():
    """
    Compute the transition matrix in advance to speed up the GPI algorithm.
    """
    nt, nx, ny, ntheta = 100, 21, 21, 40
    nv, nw = 6, 11

    @ray.remote
    def worker(start, end):
        local_matrix = np.empty((end-start, nx, ny, ntheta, nv, nw, 8, 4), dtype=np.float16)

        t_indices = np.arange(start, end)
        ex_indices = np.arange(nx)
        ey_indices = np.arange(ny)
        eth_indices = np.arange(ntheta)
        v_indices = np.arange(nv)
        w_indices = np.arange(nw)

        ex_space_arr = ex_space[0][ex_indices]
        ey_space_arr = ey_space[0][ey_indices]
        eth_space_arr = eth_space[0][eth_indices]
        v_space_arr = v_space[0][v_indices]
        w_space_arr = w_space[w_indices]

        for t in range(end-start):
            ref_cur = lissajous(t)
            ref_next = lissajous(t + 1)
            
            error_states = np.array(np.meshgrid(ex_space_arr, ey_space_arr, eth_space_arr)).T.reshape(-1, 3)
            controls = np.array(np.meshgrid(v_space_arr, w_space_arr)).T.reshape(-1, 2)

            for error_state in error_states:
                next_state_means = np.array([compute_next_state_mean(error_state, ref_cur, ref_next, control) for control in controls])
                neighbors_and_probs = np.array([compute_neighbors_and_probabilities(next_state_mean) for next_state_mean in next_state_means])

                idx_e = np.where((error_states == error_state).all(axis=1))[0][0]
                ex, ey, eth = np.unravel_index(idx_e, (nx, ny, ntheta))

                for idx_c, control in enumerate(controls):
                    v, w = np.unravel_index(idx_c, (nv, nw))
                    local_matrix[t, ex, ey, eth, v, w] = neighbors_and_probs[idx_c]

        return local_matrix

    num_workers = 32
    jobs = []
    for i in range(num_workers):
        start = i * (nt // num_workers)
        end = (i + 1) * (nt // num_workers)
        jobs.append(worker.remote(start, end))

    results = ray.get(jobs)
    ray.shutdown()

    transition_matrix = np.empty((nt, nx, ny, ntheta, nv, nw, 8, 4), dtype=np.float16)
    for j in range(num_workers):
        start = j * (nt // num_workers)
        end = (j + 1) * (nt // num_workers)
        # print(results[j])
        transition_matrix[start:end] = results[j]


    return transition_matrix


transition_matrix_file = '/home/mmkr/Documents/ECE276B/ECE276B_PR3/starter_code/transition_matrix_f16.npy'  # Specify the file path to save the array

if not os.path.exists(transition_matrix_file):
    with open(transition_matrix_file, 'w') as file:
        pass

transition_matrix = compute_transition_matrix()
np.save(transition_matrix_file, transition_matrix)  # Save the array to the file
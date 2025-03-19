from dataclasses import dataclass
import numpy as np
from value_function import ValueFunction
import utils
import ray


@dataclass
class GpiConfig:
    traj: callable
    obstacles: np.ndarray
    ex_space: np.ndarray
    ey_space: np.ndarray
    eth_space: np.ndarray
    v_space: np.ndarray
    w_space: np.ndarray
    Q: np.ndarray
    q: float
    R: np.ndarray
    gamma: float
    num_evals: int  # number of policy evaluations in each iteration
    collision_margin: float
    V: ValueFunction  # your value function implementation
    output_dir: str
    # used by feature-based value function
    v_ex_space: np.ndarray
    v_ey_space: np.ndarray
    v_etheta_space: np.ndarray
    v_alpha: float
    v_beta_t: float
    v_beta_e: float
    v_lr: float
    v_batch_size: int  # batch size if GPU memory is not enough


class GPI:
    def __init__(self, config: GpiConfig):
        self.config = config
        self.policy = np.zeros((config.V.T, len(config.ex_space), len(config.ey_space), len(config.eth_space), 2), dtype=int)
        self.transition_matrix = np.load('/home/mmkr/Documents/ECE276B/ECE276B_PR3/starter_code/transition_matrix_f16.npy',allow_pickle=True)
        self.stage_costs = np.load('/home/mmkr/Documents/ECE276B/ECE276B_PR3/starter_code/stage_cost_f16.npy',allow_pickle=True)
        self.init_value_function()
        ray.init(ignore_reinit_error=True)

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        ex, ey, etheta = cur_state - cur_ref_state
        state_idx = self.state_metric_to_index(np.array([ex, ey, etheta]))
        action_indices = self.policy[t, state_idx[0], state_idx[1], state_idx[2]]
        return self.control_index_to_metric(action_indices[0], action_indices[1])

    def state_metric_to_index(self, metric_state: np.ndarray) -> tuple:
        """
        Convert the metric state to grid indices according to your descretization design.
        Args:
            metric_state (np.ndarray): metric state
        Returns:
            tuple: grid indices
        """
        ex_idx = np.digitize(metric_state[0], self.config.ex_space, right=True) - 1
        ey_idx = np.digitize(metric_state[1], self.config.ey_space, right=True) - 1
        etheta_idx = np.digitize(metric_state[2], self.config.eth_space, right=True) - 1
        return (ex_idx, ey_idx, etheta_idx)

    def state_index_to_metric(self, state_index: tuple) -> np.ndarray:
        """
        Convert the grid indices to metric state according to your descretization design.
        Args:
            state_index (tuple): grid indices
        Returns:
            np.ndarray: metric state
        """
        ex = self.config.ex_space[state_index[0]]
        ey = self.config.ey_space[state_index[1]]
        etheta = self.config.eth_space[state_index[2]]
        return np.array([ex, ey, etheta])

    def control_metric_to_index(self, control_metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            control_metric: [2, N] array of controls in metric space
        Returns:
            [N, ] array of indices in the control space
        """
        v_idx = np.digitize(control_metric[0], self.config.v_space, right=True) - 1
        w_idx = np.digitize(control_metric[1], self.config.w_space, right=True) - 1
        return (v_idx, w_idx)

    def control_index_to_metric(self, v: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            v: [N, ] array of indices in the v space
            w: [N, ] array of indices in the w space
        Returns:
            [2, N] array of controls in metric space
        """
        return np.array([self.config.v_space[v], self.config.w_space[w]])

    # def compute_transition_matrix(self):
    #     """
    #     Compute the transition matrix in advance to speed up the GPI algorithm.
    #     """
    #     nt, nx, ny, ntheta = 100, len(self.config.ex_space), len(self.config.ey_space), len(self.config.eth_space)
    #     nv, nw = len(self.config.v_space), len(self.config.w_space)
    #     transition_matrix = np.zeros((nt, nx, ny, ntheta, nv, nw, 8, 4))

    #     @ray.remote
    #     def worker(start, end):
    #         local_matrix = np.zeros((nt, nx, ny, ntheta, nv, nw, 8, 4))
    #         for t in range(start, end):
    #             for ex in range(nx):
    #                 for ey in range(ny):
    #                     for eth in range(ntheta):
    #                         for v in range(nv):
    #                             for w in range(nw):
    #                                 state = np.array([self.config.ex_space[ex], self.config.ey_space[ey], self.config.eth_space[eth]])
    #                                 control = np.array([self.config.v_space[v], self.config.w_space[w]])
    #                                 next_state_mean = self.compute_next_state_mean(state, control)
    #                                 neighbors = self.get_neighbors(next_state_mean)
    #                                 probabilities = self.compute_transition_probabilities(next_state_mean, neighbors)
    #                                 local_matrix[t, ex, ey, eth, v, w] = neighbors, probabilities
    #         return local_matrix

    #     num_workers = 8
    #     jobs = []
    #     for i in range(num_workers):
    #         start = i * (nt // num_workers)
    #         end = (i + 1) * (nt // num_workers)
    #         jobs.append(worker.remote(start, end))

    #     results = ray.get(jobs)
    #     for result in results:
    #         transition_matrix += result

    #     return transition_matrix

    # def compute_stage_costs(self):
    #     """
    #     Compute the stage costs in advance to speed up the GPI algorithm.
    #     """
    #     nt, nx, ny, ntheta = len(self.config.traj), len(self.config.ex_space), len(self.config.ey_space), len(self.config.eth_space)
    #     nv, nw = len(self.config.v_space), len(self.config.w_space)
    #     stage_costs = np.zeros((nt, nx, ny, ntheta, nv, nw))

    #     @ray.remote
    #     def worker(start, end):
    #         local_costs = np.zeros((nt, nx, ny, ntheta, nv, nw))
    #         for t in range(start, end):
    #             for ex in range(nx):
    #                 for ey in range(ny):
    #                     for eth in range(ntheta):
    #                         for v in range(nv):
    #                             for w in range(nw):
    #                                 error_state = np.array([self.config.ex_space[ex], self.config.ey_space[ey], self.config.eth_space[eth]])
    #                                 control = np.array([self.config.v_space[v], self.config.w_space[w]])
    #                                 local_costs[t, ex, ey, eth, v, w] = self.compute_stage_cost(error_state, control)
    #         return local_costs

    #     num_workers = 8
    #     jobs = []
    #     for i in range(num_workers):
    #         start = i * (nt // num_workers)
    #         end = (i + 1) * (nt // num_workers)
    #         jobs.append(worker.remote(start, end))

    #     results = ray.get(jobs)
    #     for result in results:
    #         stage_costs += result

    #     return stage_costs

    def init_value_function(self):
        """
        Initialize the value function.
        """
        self.config.V.values.fill(0)

    def evaluate_value_function(self):
        """
        Evaluate the value function. Implement this function if you are using a feature-based value function.
        """
        # self.value_function.evaluate()
        pass

    @utils.timer
    def policy_improvement(self):
        T, ex_len, ey_len, etheta_len = self.config.V.values.shape
        nv, nw = len(self.config.v_space), len(self.config.w_space)
        
        for t in range(T):
            min_cost = np.full((ex_len, ey_len, etheta_len), np.inf)
            best_action = np.zeros((ex_len, ey_len, etheta_len, 2), dtype=int)
            
            for v_idx in range(nv):
                for w_idx in range(nw):
                    stage_cost = self.stage_costs[t, :, :, :, v_idx, w_idx]
                    
                    for k in range(8):
                        next_states = self.transition_matrix[t, :, :, :, v_idx, w_idx, k, :3]
                        next_state_idx = np.apply_along_axis(self.state_metric_to_index, 3, next_states)
                        transition_prob = self.transition_matrix[t, :, :, :, v_idx, w_idx, k, 3]
                        
                        next_value = self.config.V.values[t + 1, next_state_idx[..., 0], next_state_idx[..., 1], next_state_idx[..., 2]]
                        total_cost = stage_cost + self.config.gamma * transition_prob * next_value
                        
                        mask = total_cost < min_cost
                        min_cost[mask] = total_cost[mask]
                        best_action[mask] = self.control_index_to_metric(v_idx, w_idx)
            
            self.policy[t] = best_action

    @utils.timer
    def policy_evaluation(self):
        for _ in range(self.config.num_evals):
            new_values = np.copy(self.config.V.values)
            T, ex_len, ey_len, etheta_len = self.config.V.values.shape
            nv, nw = len(self.config.v_space), len(self.config.w_space)
            
            for t in range(T):
                min_cost = np.full((ex_len, ey_len, etheta_len), np.inf)
                
                for v_idx in range(nv):
                    for w_idx in range(nw):
                        stage_cost = self.stage_costs[t, :, :, :, v_idx, w_idx]
                        
                        for k in range(8):
                            next_states = self.transition_matrix[t, :, :, :, v_idx, w_idx, k, :3]
                            next_state_idx = np.apply_along_axis(self.state_metric_to_index, 3, next_states)
                            transition_prob = self.transition_matrix[t, :, :, :, v_idx, w_idx, k, 3]
                            
                            next_value = self.config.V.values[t + 1, next_state_idx[..., 0], next_state_idx[..., 1], next_state_idx[..., 2]]
                            total_cost = stage_cost + self.config.gamma * transition_prob * next_value
                            
                            min_cost = np.minimum(min_cost, total_cost)
                
                new_values[t] = min_cost
            
            self.config.V.values = new_values
    
    def compute_policy(self, num_iters: int) -> None:
        for n in range(num_iters):
            print(n)
            self.policy_evaluation()
            self.policy_improvement()


    def compute_neighbors_and_probabilities(self,t,state,control,dt=0.1):
        
        ref_cur = self.config.traj(t)
        ref_next = self.config.traj(t+1)
        g = np.array([
            [dt * np.cos(state[2] + ref_cur[2]), 0],
            [dt * np.sin(state[2] + ref_cur[2]), 0],
            [0, dt]
        ])
        
        # Calculate the new state
        next_state_mean = state + np.dot(g, control.T) + np.array(ref_cur) - np.array(ref_next)
        
        std = np.array([0.04, 0.04, 0.004])
        offsets = np.array([[-1, -1, 0], [-1, 0, 0], [-1, 1, 0],
                            [0, -1, 0], [0, 1, 0],
                            [1, -1, 0], [1, 0, 0], [1, 1, 0]])
        neighbors = next_state_mean + offsets

        probabilities = self.gaussian_likelihood(neighbors, next_state_mean, std)
        probabilities /= probabilities.sum()

        return np.hstack((neighbors, probabilities[:, np.newaxis]))

    def gaussian_likelihood(self,x, mean, std):
        # Handle multiple rows in x
        return np.exp(-0.5 * np.sum(((x - mean) / std) ** 2, axis=1)) / (np.sqrt(2 * np.pi) * np.prod(std))


    def compute_next_state_mean(self,t,e, u, dt=0.1):
        # Create the g matrix
        ref_cur = self.config.traj(t)
        ref_next = self.config.traj(t+1)
        g = np.array([
            [dt * np.cos(e[2] + ref_cur[2]), 0],
            [dt * np.sin(e[2] + ref_cur[2]), 0],
            [0, dt]
        ])
        
        # Calculate the new state
        return e + np.dot(g, u.T) + np.array(ref_cur) - np.array(ref_next)
        

    # def compute_next_state_mean(self, state, control):
    #     delta = 0.1  # Discretization time step
    #     theta = state[2]
    #     G = np.array([[delta * np.cos(theta), 0],
    #                   [delta * np.sin(theta), 0],
    #                   [0, delta]])
    #     return state + G @ control

    # def get_neighbors(self, next_state_mean):
    #     """
    #     Get the neighboring states for the given state.
    #     """
    #     neighbors = []
    #     for i in range(-1, 2):
    #         for j in range(-1, 2):
    #             for k in range(-1, 2):
    #                 if i == 0 and j == 0 and k == 0:
    #                     continue
    #                 neighbors.append(next_state_mean + np.array([i, j, k]))
    #     return neighbors

    # def compute_transition_probabilities(self, next_state_mean, neighbors):
    #     probabilities = []
    #     for neighbor in neighbors:
    #         probabilities.append(self.gaussian_likelihood(neighbor, next_state_mean, [0.04, 0.04, 0.004]))
    #     return np.array(probabilities) / np.sum(probabilities)

    # def gaussian_likelihood(self, x, mean, std):
    #     return np.exp(-0.5 * np.sum(((x - mean) / std) ** 2)) / (np.sqrt(2 * np.pi) * np.prod(std))

    # def compute_stage_cost(self, error_state, control):
    #     p_error = error_state[:2]
    #     theta_error = error_state[2]
    #     return p_error.T @ self.config.Q @ p_error + self.config.q * (1 - np.cos(theta_error)) ** 2 + control.T @ self.config.R @ control

    # def compute_error_state(self, cur_state, cur_ref_state):
    #     """
    #     Compute the error state.
    #     """
    #     error_state = cur_state - cur_ref_state
    #     return error_state
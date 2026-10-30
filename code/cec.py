import casadi as ca
import numpy as np
import utils

class CEC:

    def __init__(self, T=15, dt=0.5, gamma=0.9, Q=np.array([[400,0],[0,20]]), q=10, R=np.array([[20,0],[0,2]]),
                 obstacles=((-2,-2,0.5),(1,2,0.5)), margin=0.15) -> None:
        self.T = T
        self.dt = dt
        self.gamma = gamma
        self.Q = Q
        self.q = q
        self.R = R

        # The NLP is built once; the current state and the reference over the horizon are parameters
        u = ca.SX.sym('u', 2 * self.T)          # control inputs for T time steps
        x0 = ca.SX.sym('x0', 3)
        refs = ca.SX.sym('refs', 3 * self.T)    # reference for each predicted state
        objective = 0
        constraints = []
        Xk = x0
        for k in range(self.T):
            uk = u[2*k:2*k+2]
            Xk = utils.car_next_state_casadi(self.dt, Xk, uk)
            objective += self.cost_function(Xk - refs[3*k:3*k+3], uk, k)
            # Stay outside each obstacle, plus a margin for the motion noise (sigma = 0.04 m per step)
            for ox, oy, r in obstacles:
                constraints.append((Xk[0] - ox)**2 + (Xk[1] - oy)**2)

        self.lbg = np.array([(r + margin)**2 for _ in range(self.T) for _, _, r in obstacles])
        self.ubg = np.full(len(constraints), np.inf)
        self.lbx = np.tile([utils.v_min, utils.w_min], self.T)
        self.ubx = np.tile([utils.v_max, utils.w_max], self.T)
        nlp = {'x': u, 'p': ca.vertcat(x0, refs), 'f': objective, 'g': ca.vertcat(*constraints)}
        self.solver = ca.nlpsol("S", "ipopt", nlp, {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': False})
        self.u_prev = np.zeros(2 * self.T)

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step (iteration index, not seconds)
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        # The state after k+1 steps should reach the reference of step t+k (main.py scores the
        # next state against the current reference)
        refs = np.concatenate([utils.lissajous(t + k) for k in range(self.T)])
        # Warm start from the previous plan, shifted one step
        x0 = np.concatenate([self.u_prev[2:], self.u_prev[-2:]])
        sol = self.solver(x0=x0, p=np.concatenate([cur_state, refs]),
                          lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        self.u_prev = np.array(sol["x"]).flatten()
        return self.u_prev[0:2]

    def cost_function(self, e, u, k):
        p_cost = ca.mtimes([e[0:2].T, self.Q, e[0:2]])
        theta_cost = self.q * (1 - ca.cos(e[2]))**2
        u_cost = ca.mtimes([u.T, self.R, u])
        return self.gamma**k*(p_cost + theta_cost + u_cost)

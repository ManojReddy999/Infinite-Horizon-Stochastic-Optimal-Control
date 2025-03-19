import casadi as ca
import numpy as np
import utils

class CEC:

    def __init__(self, T=15, dt=0.5, gamma=0.9, Q=np.array([[400,0],[0,20]]), q=10, R=np.array([[20,0],[0,2]])) -> None:
        self.T = T
        self.dt = dt
        self.gamma = gamma
        self.Q = Q
        self.q = q
        self.R = R
        self.obs1 = ca.MX([-2,-2,0.5])
        self.obs2 = ca.MX([1,2,0.5])

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
        # TODO: define optimization variables
        u = ca.MX.sym('u', 2 * self.T)  # Control inputs for T time steps
        e = ca.MX.sym('e', 3 * (self.T + 1))
        # TODO: define optimization constraints and optimization objective
        
        Xk = cur_state
        e0 = ca.MX(cur_state - cur_ref_state)
        objective = self.cost_function(e0,u[0:2],0)
        print(self.collision_check(cur_state[:2]))
        constraints = []
        constraints.append(e[0:3]-e0)

        for k in range(self.T):
            uk = u[2*k:2*k+2]
            Xk_next = utils.car_next_state_casadi(self.dt, Xk, uk)
            
            # if self.collision_check(Xk_next[:2]):    
            constraints.append(ca.sqrt((Xk_next[0] - self.obs1[0])**2 + (Xk_next[1] - self.obs1[1])**2) - self.obs1[2]>0)
            constraints.append(ca.sqrt((Xk_next[0] - self.obs2[0])**2 + (Xk_next[1] - self.obs2[1])**2) - self.obs2[2]>2)
            
            Xk_next[2] = self.normalize_angle(Xk_next[2]) 
            ref_next = utils.lissajous(t+k+1)
            ek_next = Xk_next - ref_next
            ek_next[2] = self.normalize_angle(ek_next[2]) 
            objective += self.cost_function(ek_next, uk, k)
            constraints.append(e[3*(k+1):3*(k+2)] == ek_next)  # This stores the propagated state
            
            Xk = Xk_next

        objective -= uk.T @ self.R @ uk



        # TODO: define optimization solver
        nlp = {'x': ca.vertcat(u,e), 'f': objective, 'g': ca.vertcat(*constraints)}
        solver = ca.nlpsol("S", "ipopt", nlp)
        sol = solver(
            x0=ca.DM(np.zeros(2 * self.T + 3 * (self.T + 1))),  # TODO: initial guess
            lbx=ca.DM(np.concatenate([[utils.v_min, utils.w_min] * self.T, [-np.inf] * 3 * (self.T + 1)])), # TODO: lower bound on optimization variables
            ubx=ca.DM(np.concatenate([[utils.v_max, utils.w_max] * self.T, [np.inf] * 3 * (self.T + 1)])), # TODO: upper bound on optimization variables
            lbg=ca.DM(np.zeros(2 * self.T + 3 * (self.T+1))), # TODO: lower bound on optimization constraints
            ubg=ca.DM(np.zeros(2 * self.T + 3 * (self.T+1))), # TODO: upper bound on optimization constraints
        )
        x = sol["x"]  # get the solution

        # TODO: extract the control input from the solution
        u = np.array(x[0:2])
        return u
    
    # def cost_function(self, x, u, k):
    #     p_cost = ca.mtimes([x[0:2].T, self.Q, x[0:2]])
    #     theta_cost = self.q * (1 - ca.cos(x[2]))**2
    #     u_cost = ca.mtimes([u, self.R, u.T])
    #     return self.gamma**k*(p_cost + theta_cost + u_cost)

    def cost_function(self, x, u, k):
        p_cost = x[0:2].T @ self.Q @ x[0:2]
        theta_cost = self.q * (1 - np.cos(x[2]))**2
        u_cost = u.T @ self.R @ u
        return self.gamma**k*(p_cost + theta_cost + u_cost)
    
    def G(self, dt, e, ref_cur,ref_next, u):
        g = ca.vertcat(
            ca.horzcat(dt * ca.cos(e[2] + ref_cur[2]), 0),
            ca.horzcat(dt * ca.sin(e[2] + ref_cur[2]), 0),
            ca.horzcat(0, dt)
        )
        return e + ca.mtimes(g, u.T) + (ref_cur - ref_next)
    
    def collision_check(self, x):
        return ca.logic_or(ca.norm_2(x-self.obs1[:2]) < self.obs1[2], ca.norm_2(x-self.obs2[:2]) < self.obs2[2])
        
    def normalize_angle(self, angle):
        """ Normalize angle to be within the interval [-pi, pi]. """
        return angle - 2 * ca.pi * ca.floor((angle + ca.pi) / (2 * ca.pi))
        
    

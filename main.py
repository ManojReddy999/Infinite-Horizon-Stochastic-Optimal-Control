from time import time
import numpy as np
import utils
import cec
from gpi import GPI, GpiConfig
from value_function import GridValueFunction, FeatureValueFunction


def main():
    # Obstacles in the environment
    obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
    # Params
    traj = utils.lissajous
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    cur_iter = 0

        # Set up GPI configuration
    # config = GpiConfig(
    #     traj=traj,
    #     obstacles=obstacles,
    #     ex_space=np.linspace(-3, 3, 21),  # Example discretization, change as needed
    #     ey_space=np.linspace(-3, 3, 21),
    #     eth_space=np.linspace(-np.pi, np.pi, 40),
    #     v_space=np.linspace(0, 1, 6),  # Example control space, change as needed
    #     w_space=np.linspace(-1, 1, 11),
    #     Q=np.eye(2),  # Example weight matrix for state cost
    #     q=1.0,  # Example stage cost
    #     R=np.eye(2),  # Example weight matrix for control cost
    #     gamma=0.9,  # Example discount factor
    #     num_evals=10,  # Number of policy evaluations in each iteration
    #     collision_margin=0.1,  # Example collision margin
    #     V=GridValueFunction(100,np.linspace(-3, 3, 21), np.linspace(-3, 3, 21), np.linspace(-np.pi, np.pi, 40)),  # Example value function
    #     output_dir='./output',  # Example output directory
    #     v_ex_space=np.linspace(-3, 3, 21),
    #     v_ey_space=np.linspace(-3, 3, 21),
    #     v_etheta_space=np.linspace(-np.pi, np.pi, 40),
    #     v_alpha=0.1,  # Example alpha value for feature-based value function
    #     v_beta_t=0.1,  # Example beta_t value for feature-based value function
    #     v_beta_e=0.1,  # Example beta_e value for feature-based value function
    #     v_lr=0.01,  # Example learning rate for feature-based value function
    #     v_batch_size=32  # Example batch size
    # )


    # Initialize GPI controller
    # gpi_controller = GPI(config)
    # gpi_controller.compute_policy(10)

    # Main loop
    while cur_iter * utils.time_step < utils.sim_time:
        t1 = time()
        # Get reference state
        cur_time = cur_iter * utils.time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        # control = utils.simple_controller(cur_state, cur_ref)

        control_cec = cec.CEC()
        control = control_cec(cur_time,cur_state,cur_ref)

        # control = gpi_controller(cur_iter, cur_state, cur_ref)

        print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = utils.time()
        print(cur_iter)
        print(t2 - t1)
        times.append(t2 - t1)
        cur_err = cur_state - cur_ref
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans = error_trans + np.linalg.norm(cur_err[:2])
        error_rot = error_rot + np.abs(cur_err[2])
        print(cur_err, error_trans, error_rot)
        print("======================")
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print("\n\n")
    print("Total time: ", main_loop_time - main_loop)
    print("Average iteration time: ", np.array(times).mean() * 1000, "ms")
    print("Final error_trains: ", error_trans)
    print("Final error_rot: ", error_rot)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, save=True)


if __name__ == "__main__":
    main()


"""Compare the receding-horizon CEC with the provided P controller over noisy runs.

Each run is main.py's loop without plotting: 120 s (240 steps) with motion noise, the
error scored like main.py (next state against the current reference).

usage: python evaluate.py [runs]
"""
import sys
from time import perf_counter
import numpy as np
import utils
import cec

obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def run(controller, seed):
    np.random.seed(seed)
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    errors, clearances, times = [], [], []
    for cur_iter in range(int(utils.sim_time / utils.time_step)):
        cur_ref = utils.lissajous(cur_iter)
        t1 = perf_counter()
        control = controller(cur_iter, cur_state, cur_ref)
        times.append(perf_counter() - t1)
        cur_state = utils.car_next_state(utils.time_step, cur_state, np.asarray(control, dtype=float).flatten(), noise=True)
        errors.append(np.linalg.norm(cur_state[:2] - np.array(cur_ref[:2])))
        clearances.append(min(np.hypot(cur_state[0] - ox, cur_state[1] - oy) - r for ox, oy, r in obstacles))
    clearances = np.array(clearances)
    return np.mean(errors), np.max(errors), int((clearances < 0).sum()), clearances.min(), np.median(times)


controllers = {
    'P controller (utils.simple_controller)': lambda: (lambda it, s, r: utils.simple_controller(s, r)),
    'Receding-horizon CEC (cec.py)': lambda: cec.CEC(obstacles=obstacles),
}
print(f'{runs} runs of {utils.sim_time} s each, motion noise sigma = {utils.sigma.tolist()}\n')
for name, make in controllers.items():
    rows = np.array([run(make(), seed) for seed in range(runs)])
    print(name)
    print(f'  mean position error: {rows[:, 0].mean():.3f} m (runs range {rows[:, 0].min():.3f}-{rows[:, 0].max():.3f})')
    print(f'  worst position error: {rows[:, 1].max():.3f} m')
    print(f'  steps inside an obstacle: {int(rows[:, 2].sum())} of {runs * int(utils.sim_time / utils.time_step)}'
          f' ({int((rows[:, 2] > 0).sum())} of {runs} runs), closest approach {rows[:, 3].min():+.3f} m')
    print(f'  median time per control step: {1000 * np.median(rows[:, 4]):.1f} ms\n')

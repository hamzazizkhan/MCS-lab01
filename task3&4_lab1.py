import numpy as np
import matplotlib.pyplot as plt
# discrete time step
dt = 0.01

# simulation time range
time = np.arange(1.0, 4.0 + dt, dt)

# second-order system initial conditions [y1, y2] @ t = 1
y0 = np.array([0, 1])


def ode_system(_t, _y):
    """
    system of first order differential equations
    _t: discrete time step value
    _y: state vector [y1, y2]
    """
    return np.array([_y[1], -_t * _y[1] + (2 / _t) * _y[0]])

# runge-kutta fourth-order numerical integration
def rk4(func, tk, _yk, _dt=0.01, **kwargs):
    """
    single-step fourth-order numerical integration (RK4) method
    func: system of first order ODEs
    tk: current time step
    _yk: current state vector [y1, y2, y3, ...]
    _dt: discrete time step size
    **kwargs: additional parameters for ODE system
    returns: y evaluated at time k+1
    """

    # evaluate derivative at several stages within time interval
    f1 = func(tk, _yk, **kwargs)
    f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), **kwargs)
    f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), **kwargs)
    f4 = func(tk + _dt, _yk + (f3 * _dt), **kwargs)

    # return an average of the derivative over tk, tk + dt
    return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)

# ==============================================================
# propagate state

# simulation results
state_history = []

# initialize yk
yk = y0

# intialize time
t = 0

# approximate y at time t
for t in time:
    state_history.append(yk)
    yk = rk4(ode_system, t, yk, dt)

# convert list to numpy array
state_history = np.array(state_history)


# =========== part 2 ===========

def lorenz(_t, _y, sigma=10, beta=(8 / 3), rho=28):
    """
    lorenz chaotic differential equation: dy/dt = f(t, y)
    _t: time tk to evaluate system
    _y: 3D state vector [x, y, z]
    sigma: constant related to Prandtl number
    beta: geometric physical property of fluid layer
    rho: constant related to the Rayleigh number
    return: [x_dot, y_dot, z_dot]
    """
    return np.array([
        sigma * (_y[1] - _y[0]),
        _y[0] * (rho - _y[2]) - _y[1],
        (_y[0] * _y[1]) - (beta * _y[2]),
    ])

# ==============================================================
# simulation harness

# discrete time step size
dt = 0.01

# simulation time range
time = np.arange(0.0, 8.0, dt)

# lorenz initial conditions (x, y, z) at t = 0 so that orbit is in the basin of attraction of lorenz attractor
y0 = np.array([10, 10, 10])

# ==============================================================
# propagate state

# simulation results
state_history = []

# initialize yk
yk = y0

# intialize time
t = 0

# iterate over time
for t in time:
    # save current state
    state_history.append(yk)

    # update state variables yk to yk+1
    yk = rk4(lorenz, t, yk, dt)

# convert list to numpy array
state_history = np.array(state_history)


# plottting in 3D
x = [pt[0] for pt in state_history]
y = [pt[1] for pt in state_history]
z = [pt[2] for pt in state_history]

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(x, y, z, 'gray')
plt.show()

# differentiate lorenz system forward in time
def odeint(time,y0):
    yk=y0
    yk = rk4(lorenz, time, yk, dt)
    return np.array(yk)



# perturbation vector
d0 = 1e-8 * np.ones(3)

dist=[]
lyapunov_sum = []
lyapunov_sum.append(0)

# perturbed state history
perturbed = []

y0_perturbed = state_history[0][2] + d0
perturbed.append(y0_perturbed)

# advance orbit B
state_pert = odeint(time[0], y0_perturbed)
perturbed.append(state_pert)
# calculate seperation
d1 = np.sqrt(np.abs(state_history[1][0] - state_pert[0]) ** 2 + np.abs(state_history[1][1] - state_pert[1]) ** 2 + np.abs(state_history[1][2] - state_pert[2]) ** 2)
ln = np.log(d1 / 1e-8)
lyapunov_sum.append(ln)



for i in range(2,len(time)):

    # ensure orbit B is seperated by 1e-8 and is in same direction as seperation (d1)
    xb0 = state_history[i-1][0] + (1e-8*(perturbed[i-1][0]-state_history[i-1][0])) /d1
    yb0 = state_history[i-1][1] + (1e-8*(perturbed[i-1][1]-state_history[i-1][1])) /d1
    zb0 = state_history[i-1][2] + (1e-8*(perturbed[i-1][2]-state_history[i-1][2])) /d1
    y0_perturbed = np.array([xb0, yb0, zb0])

    # advance orbit B
    state_pert = odeint(time[i-1], y0_perturbed)

    perturbed.append(state_pert)
    # calc seperation
    d1 = np.sqrt(np.abs(state_history[i][0] - state_pert[0]) ** 2 + np.abs(
            state_history[i][1] - state_pert[1]) ** 2 + np.abs(state_history[i][2] - state_pert[2]) ** 2)
    ln = np.log(d1 / 1e-8)
    dist.append(d1)
    lyapunov_sum.append(ln)


x_ax = [time[i] for i in range(800)]
y_ax = [lyapunov_sum[i] for i in range(800)]
print('average',(sum(y_ax)/800)/dt)


















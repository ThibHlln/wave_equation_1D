r"""wave motion in 1D:

        linearised partial differential equation:
            \frac{\partial^2 u}{\partial t^2} = \gamma^2 \frac{\partial^2 u}{\partial x^2}, \; x \in (a, b)

            u(x, t) is the displacement along direction y
            \gamma is a constant

        boundary conditions: fixed at ends
            u(a, t) = 0
            u(b, t) = 0

        initial conditions: initially at rest
            u(x, 0) = I(x)  a given inflexion
            u'(x, 0) = 0  no initial velocity

    finite difference approximation:

        discretisation in space and time:
            x_i = (i - 1) \Delta x, \; i = 1, ..., nx
            t_l = l \Delta t, \; l = 0, 1, ..., nt

        central difference approximation in space:
            \frac{\partial^2 u}{\partial x^2}(x_i, t_l) \approx
                \frac{ u_{i-1}^{l} - 2 u_{i}^{l} + u_{i+1}^{l} }
                     { \Delta x^2 }

        central difference approximation in time
            \frac{\partial^2 u}{\partial t^2}(x_i, t_l) \approx
                \frac{ u_{i}^{l-1} - 2 u_{i}^{l} + u_{i}^{l+1} }
                     { \Delta t^2 }

        discretised equation:
            u_{i}^{l+1} = 2 u_{i}^{l} - u_{i}^{l-1} + c^2 (u_{i-1}^{l} - 2u_{i}^{l} + u_{i+1}^{l})

            where c is the Courant–Friedrichs–Lewy number
                c = \gamma \frac{\Delta t}{\Delta x}

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


def set_initial_conditions(u, um, delta_x, nx, c, u_max=0.05):
    # set the initial displacement u(x, 0)
    x = np.arange(nx) * delta_x
    u[:] = np.where(x < 0.7,
                    (u_max / 0.7) * x,
                    (u_max / 0.3) * (1 - x))

    # use modified stencil to determine initial um
    um[1:-1] = u[1:-1] + 0.5 * c * c * (u[2:] - 2 * u[1:-1] + u[0:-2])


def solve(up, u, um, nx, nt, c):
    # create saving array
    sol = np.zeros((nx, nt), np.float64)

    # save initial conditions
    sol[:, 0] = u[:]

    for t in range(1, nt):
        # inner points
        up[1:-1] = 2 * u[1:-1] - um[1:-1] + c * c * (u[2:] - 2 * u[1:-1] + u[0:-2])
        # boundary conditions
        up[0] = 0
        up[-1] = 0
        # swap storage arrays
        um[:] = u  # u at t becomes u at t-1
        u[:] = up  # u at t+1 becomes u at t
        # save solution at t
        sol[:, t] = up

    return sol


if __name__ == '__main__':
    # set up parameters
    nx_ = 21  # number of points along x
    t_stop_ = 0.5  # stopping time for iterations
    c_ = 0.3  # Courant number

    # infer space and time resolutions
    delta_x_ = 1 / (nx_ - 1)
    delta_t_ = c_ * delta_x_

    # infer number of time steps
    nt_ = int(t_stop_ // delta_t_)

    # create storing arrays
    up_ = np.zeros(nx_, dtype=np.float64)  # u(:, l+1)
    u_ = np.zeros(nx_, dtype=np.float64)  # u(:, l)
    um_ = np.zeros(nx_, dtype=np.float64)  # u(:, l-1)

    # initialise
    set_initial_conditions(u_, um_, delta_x_, nx_, c_)
    # run
    sol_ = solve(up_, u_, u_, nx_, nt_, c_)

    # plot
    fig, ax = plt.subplots()

    cm = plt.get_cmap('viridis')
    c_norm = colors.Normalize(vmin=0, vmax=nt_ - 1)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

    for t_ in range(nt_):
        ax.plot(np.arange(nx_), sol_[:, t_], color=scalar_map.to_rgba(t_))

    plt.show()

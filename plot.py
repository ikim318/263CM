import warnings
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from sklearn.linear_model import BayesianRidge


# This function defines your ODE.
def ode_model(t, x, q, dqdt, a, b, c, x0):
    """ Return the derivative dx/dt at time, t, for given parameters.
        Parameters:
        -----------
        t : float
            Independent variable time.
        x : float
            Dependent variable (pressure or temperature)
        q : float
            mass injection/ejection rate.
        a : float
            mass injection strength parameter.
        b : float
            recharge strength parameter.
        x0 : float
            Ambient value of dependent variable.
        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable time.
        Notes:
        ------
        None
    """
    # equation to return the derivative of dependent variable with respect to time

    # TYPE IN YOUR TEMPERATURE ODE HERE
    dxdt = -a * q - b * (x - x0) - c * dqdt

    return dxdt


# This function loads in your data.
def load_data():
    """ Load data throughout the time period.
    Parameters:
    -----------
    Returns:
    ----------
    t_q : array-like
        Vector of times at which measurements of q were taken.
    q : array-like
        Vector of q (units)
    t_x : array-like
        Vector of times at which measurements of x were taken.
    x : array-like
        Vector of x (units)
    """
    # Load kettle data
    time, q, x = np.genfromtxt('Dataset.csv', delimiter=',', skip_header=1).T

    return time, q, x


# This function solves your ODE using Improved Euler
def solve_ode(f, t0, t1, dt, xi, pars):
    """ Solve an ODE using the Improved Euler Method.
    Parameters:
    -----------
    f : callable
        Function that returns dxdt given variable and parameter inputs.
    t0 : float
        Initial time of solution.
    t1 : float
        Final time of solution.
    dt : float
        Time step length.
    xi : float
        Initial value of solution.
    pars : array-like
        List of parameters passed to ODE function f.
    Returns:
    --------
    t : array-like
        Independent variable solution vector.
    x : array-like
        Dependent variable solution vector.
    Notes:
    ------
    Assume that ODE function f takes the following inputs, in order:
        1. independent variable
        2. dependent variable
        3. forcing term, q
        4. all other parameters
    """

    # set an arbitrary initial value of q and dqdt for benchmark solution
    q = 1.0
    dqdt = 0
    if pars is None:
        pars = []

    # calculate the time span
    tspan = t1 - t0
    # use floor rounding to calculate the number of variables
    n = int(tspan // dt)

    # initialise the independent and dependent variable solution vectors
    x = [xi]
    t = [t0]

    # perform Improved Euler to calculate the independent and dependent variable solutions
    for i in range(n):
        f0 = f(t[i], x[i], q, dqdt, *pars)
        f1 = f(t[i] + dt, x[i] + dt * f0, q, dqdt, *pars)
        x.append(x[i] + dt * (f0 / 2 + f1 / 2))
        t.append(t[i] + dt)

    return t, x


# This function defines your ODE as a numerical function suitable for calling 'curve_fit' in scipy.
def x_curve_fitting(t, a, b, c):
    """ Function designed to be used with scipy.optimize.curve_fit which solves the ODE using the Improved Euler Method.
        Parameters:
        -----------
        t : array-like
            Independent time variable vector
        a : float
            mass injection strength parameter.
        b : float
            recharge strength parameter.
        Returns:
        --------
        x : array-like
            Dependent variable solution vector.
        """
    # model parameters
    pars = [a, b, c]

    # ambient value of dependent variable
    x0 = 1

    # time vector information
    n = len(t)
    dt = t[1] - t[0]

    # read in time and dependent variable information
    [t, x_exact] = [load_data()[0], load_data()[2]]

    # initialise x
    x = [x_exact[0]]

    # read in q data
    [t_q, q] = [load_data()[0], load_data()[1]]

    # using interpolation to find the injection rate at each point in time
    q = np.interp(t, t_q, q)
    dqdt = find_dqdt(q, t)

    # using the improved euler method to solve the ODE
    for i in range(n - 1):
        f0 = ode_model(t[i], x[i], q[i], dqdt[i], *pars, x0)
        f1 = ode_model(t[i] + dt, x[i] + dt * f0, q[i], dqdt[i], *pars, x0)
        x.append(x[i] + dt * (f0 / 2 + f1 / 2))

    return x


# This function calls 'curve_fit' to improve your parameter guess.
def x_pars(pars_guess):
    """ Uses curve fitting to calculate required parameters to fit ODE equation
    Parameters
    ----------
    pars_guess : array-like
        Initial parameters guess
    Returns
    -------
    pars : array-like
           Array consisting of a: mass injection strength parameter, b: recharge strength parameter
    """
    # read in time and dependent variable data
    [t_exact, x_exact] = [load_data()[0], load_data()[2]]

    # finding model constants in the formulation of the ODE using curve fitting
    # optimised parameters (pars) and covariance (pars_cov) between parameters
    pars, pars_cov = curve_fit(x_curve_fitting, t_exact, x_exact, pars_guess)

    return pars, pars_cov


# This function solves your ODE using Improved Euler for a future prediction with new q
def solve_ode_prediction(f, t0, t1, dt, xi, q, a, b, c, dqdt, x0):
    """ Solve the pressure prediction ODE model using the Improved Euler Method.
    Parameters:
    -----------
    f : callable
        Function that returns dxdt given variable and parameter inputs.
    t0 : float
        Initial time of solution.
    t1 : float
        Final time of solution.
    dt : float
        Time step length.
    xi : float
        Initial value of solution.
    a : float
        mass injection strength parameter.
    b : float
        recharge strength parameter.
    x0 : float
        Ambient value of solution.
    Returns:
    --------
    t : array-like
        Independent variable solution vector.
    x : array-like
        Dependent variable solution vector.
    Notes:
    ------
    Assume that ODE function f takes the following inputs, in order:
        1. independent variable
        2. dependent variable
        3. forcing term, q
        4. all other parameters
    """
    # finding the number of time steps
    tspan = t1 - t0
    n = int(tspan // dt)

    # initialising the time and solution vectors
    x = [xi]
    t = [t0]

    # using the improved euler method to solve the pressure ODE
    for i in range(n):
        f0 = f(t[i], x[i], q, dqdt, a, b, c, x0)
        f1 = f(t[i] + dt, x[i] + dt * f0, q, dqdt, a, b, c, x0)
        x.append(x[i] + dt * (f0 / 2 + f1 / 2))
        t.append(t[i] + dt)

    return t, x


# This function plots your model over the data using your estimate for a and b
def plot_suitable():
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # read in time and pressure data
    [t, q, x_exact] = [load_data()[0], load_data()[1], load_data()[2]]
    dqdt = find_dqdt(q, t)
    # TYPE IN YOUR PARAMETER ESTIMATE FOR a AND b HERE
    a = 9.81/(12000000*0.2)
    b = (10**-13*1000*12000000)/(8.9*10**-9*np.sqrt(12000000))*a
    c = 0.025
    pars = [a, b, c]

    # solve ODE with estimated parameters and plot
    x = x_curve_fitting(t, *pars)
    ax1.plot(t, x_exact, 'k.', label='Observation')
    ax1.plot(t, x, 'r-', label='Curve Fitting Model')
    ax1.set_ylabel('Pressure (Bar)')
    ax1.set_xlabel('Time (Years)')
    ax1.legend()

    # compute the model misfit and plot
    misfit = x
    for i in range(len(x)):
        misfit[i] = x_exact[i] - x[i]
    ax2.plot(t, misfit, 'x', label='misfit', color='r')
    ax2.set_ylabel('Pressure misfit (Bar)')
    ax2.set_xlabel('Time (sec)')
    plt.axhline(y=0, color='k', linestyle='-')
    ax2.legend()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


# This function plots your model over the data using your improved model after curve fitting.
def plot_improve():
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # read in time and temperature data
    [t, x_exact] = [load_data()[0], load_data()[2]]
    # TYPE IN YOUR PARAMETER GUESS FOR a AND b HERE AS A START FOR OPTIMISATION
    a = 9.81 / (12000000 * 0.2)
    b = (10 ** -13 * 1000 * 12000000) / (8.9 * 10 ** -9 * np.sqrt(12000000)) * a
    c = 0.025
    pars_guess = [a,b,c]

    # call to find out optimal parameters using guess as start
    pars, pars_cov = x_pars(pars_guess)

    # check new optimised parameters
    print("Improved a,b,c")
    print(pars[0], pars[1], pars[2])

    # solve ODE with new parameters and plot
    x = x_curve_fitting(t, *pars)
    ax1.plot(t, x_exact, 'k.', label='Observation')
    ax1.plot(t, x, 'r-', label='Curve Fitting Model')
    ax1.set_ylabel('Pressure (Bar)')
    ax1.set_xlabel('Time (Year)')
    ax1.legend()

    # compute the model misfit and plot
    misfit = x
    for i in range(len(x)):
        misfit[i] = x_exact[i] - x[i]
    ax2.plot(t, misfit, 'x', label='misfit', color='r')
    ax2.set_ylabel('Pressure misfit (Bar)')
    ax2.set_xlabel('Time (Year)')
    plt.axhline(y=0, color='k', linestyle='-')
    ax2.legend()

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


# This function plots your model against a benchmark analytic solution.
def plot_benchmark():
    """ Compare analytical and numerical solutions via plotting.

    Parameters:
    -----------
    none

    Returns:
    --------
    none

    """
    # values for benchmark solution
    t0 = 0
    t1 = 10
    dt = 0.1

    # model values for benchmark analytic solution
    a = 1
    b = 1
    q = 1
    c = 1
    dqdt = 0
    # set ambient value to zero for benchmark analytic solution
    x0 = 0
    # set inital value to zero for benchmark analytic solution
    xi = 0

    # setup parameters array with constants
    pars = [a, b, c, x0]

    fig, plot = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))

    # Solve ODE and plot
    t, x = solve_ode(ode_model, t0, t1, dt, xi, pars)
    plot[0].plot(t, x, "bx", label="Numerical Solution")
    plot[0].set_ylabel("Pressure (Bar)")
    plot[0].set_xlabel("t")
    plot[0].set_title("Benchmark")

    # Analytical Solution
    t = np.array(t)

    #   TYPE IN YOUR ANALYTIC SOLUTION HERE
    x_analytical = -((a * q) / b) * (1 - np.e ** (-b * t)) - c * dqdt

    plot[0].plot(t, x_analytical, "r-", label="Analytical Solution")
    plot[0].legend(loc=1)

    # Plot error
    x_error = []
    for i in range(1, len(x)):
        if (x[i] - x_analytical[i]) == 0:
            x_error.append(0)
            print("check line Error Analysis Plot section")
        else:
            x_error.append((np.abs(x[i] - x_analytical[i]) / np.abs(x_analytical[i])))
    plot[1].plot(t[1:], x_error, "k*")
    plot[1].set_ylabel("Relative Error Against Benchmark")
    plot[1].set_xlabel("t")
    plot[1].set_title("Error Analysis")
    plot[1].set_yscale("log")

    # Timestep convergence plot
    time_step = np.flip(np.linspace(1 / 5, 1, 13))
    for i in time_step:
        t, x = solve_ode(ode_model, t0, t1, i, x0, pars)
        plot[2].plot(1 / i, x[-1], "kx")

    plot[2].set_ylabel(f"Pressure(t = {10})")
    plot[2].set_xlabel("1/\u0394t")
    plot[2].set_title("Timestep Convergence")

    # plot spacings
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.show()


def find_dqdt(q, t):
    dqdt = np.zeros(len(q))
    for i in range(len(q) - 1):
        dqdt[i] = q[i + 1] - q[i]
    return dqdt

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
    return -a * q - b * (x - x0) - c * dqdt


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
def x_curve_fitting(t, a, b, c, x0):
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
    pars = [a, b, c, x0]

    # time vector information
    n = len(t)
    dt = t[1] - t[0]

    # read in time and dependent variable information
    [t,q,x_e] = np.genfromtxt("TotalDataSet.csv",delimiter=',',skip_header=1).T

    # initialise x
    x = [x_e[0]]

    # using interpolation to find the injection rate at each point in time
    dqdt = find_dqdt(q)

    # using the improved euler method to solve the ODE
    for i in range(n - 1):
        f0 = ode_model(t[i], x[i], q[i], dqdt[i], *pars)
        f1 = ode_model(t[i] + dt, x[i] + dt * f0, q[i], dqdt[i], *pars)
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
    [t, x_exact] = load_data()[0], load_data()[2]
    # TYPE IN YOUR PARAMETER ESTIMATE FOR a AND b HERE
    a = 9.81 / (400000 * 0.2)
    b = (10 ** -14 * 1000 * 400000) / (8.9 * 10 ** -8 * 400) * a
    c = 0.01
    # Guess x0 is 40 for average depth of reservoir
    x0 = 40
    pars = [a, b, c, x0]

    # solve ODE with estimated parameters and plot
    x = x_curve_fitting(t, *pars)
    ax1.plot(t, x_exact, 'k.', label='Observation')
    ax1.plot(t, x, 'r-', label='Curve Fitting Model')
    ax1.set_ylabel('Pressure (Bar)')
    ax1.set_xlabel('Time (Year)')
    ax1.legend()

    # compute the model misfit and plot
    misfit = x
    sum = 0
    for i in range(len(x)):
        misfit[i] = x_exact[i] - x[i]
        sum += (misfit[i]) ** 2
    print(f"Misfit is: {sum}")
    ax2.plot(t, misfit, 'x', label='misfit', color='r')
    ax2.set_ylabel('Pressure misfit (Bar)')
    ax2.set_xlabel('Time (Year)')
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
    a = 9.81 / (16000000 * 0.2)
    b = (10 ** -14 * 1000 * 16000000) / (8.9 * 10 ** -8 * 400) * a
    c = 0.01
    # Guess x0 is 40 for average depth of reservoir
    x0 = 40
    pars_guess = [a, b, c, x0]

    # call to find out optimal parameters using guess as start
    pars, pars_cov = x_pars(pars_guess)

    # check new optimised parameters
    print("Improved a,b,c,x0")
    print(pars[0], pars[1], pars[2], pars[3])

    # solve ODE with new parameters and plot
    x = x_curve_fitting(t, *pars)
    ax1.plot(t, x_exact, 'k.', label='Observation')
    ax1.plot(t, x, 'r-', label='Curve Fitting Model')
    ax1.set_ylabel('Pressure (Bar)')
    ax1.set_xlabel('Time (Year)')
    ax1.legend()

    # compute the model misfit and plot
    misfit = x
    sum = 0
    for i in range(len(x)):
        misfit[i] = x_exact[i] - x[i]
        sum += (misfit[i]) ** 2
    print(f"Misfit is: {sum}")
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
    # set initial value to zero for benchmark analytic solution
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


def find_dqdt(q):
    # Finds dqdt between two consecutive years (since dt is 1, don't need time data)
    dqdt = np.zeros(len(q))
    for i in range(len(q) - 1):
        dqdt[i] = q[i + 1] - q[i]
    return dqdt


def plot_x_forecast():
    ''' Plot the ODE LPM model over the given data plot with different q-value scenario for predictions.
    Use a curve fitting function to accurately define the optimum parameter values.
    Parameters:
    -----------
    none
    Returns:
    --------
    none
    '''

    """
       This function plots the uncertainty of the ODE model.
       """

    #   Guess Parameters
    a = 0.001855863594537528
    b = 0.09308619728884998
    c = 0.006336076723279304
    x0 = 53.77866150176843
    pars = [a, b, c, x0]
    [t, q, x] = np.genfromtxt("TotalDataSet.csv", delimiter=',', skip_header=1).T

    # # Optimise parameters for model fit
    # pars, pars_cov = x_pars(pars_guess)

    # Store optimal values for later use

    [a, b, c, x0] = pars

    # Solve ODE and plot model
    x_e = x_curve_fitting(t, *pars)
    figa, ax1 = plt.subplots()
    ax1.plot(t, x, 'r.', label='data')
    ax1.plot(t, x_e, 'black', label='Model')

    # Remember the last time
    t_end = t[-1]

    # Create forecast time with 400 new time steps
    t1 = []
    for i in range(50):
        t1.append(i + t_end)

    # Set initial and ambient values for forecast
    xi = x_e[-1]  # Initial value of x is final value of model fit

    # Solve ODE prediction for scenario 1 (Iwi)
    q1 = 700  # heat up again
    x1 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q1, a, b, c, 0, x0)[1]
    ax1.plot(t1, x1, 'purple', label='Prediction when q = 700 (Iwi)')

    # Solve ODE prediction for scenario 2 (Homeowners)
    q2 = 900  # keep q the same
    x2 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q2, a, b, c, 0, x0)[1]
    ax1.plot(t1, x2, 'green', label='Prediction when q = 900 (Homeowners)')

    # Solve ODE prediction for scenario 3 (Geothermal Company)
    q3 = 1250  # extract at faster rate
    x3 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q3, a, b, c, 0, x0)[1]
    ax1.plot(t1, x3, 'blue', label='Prediction when q = 1250 (Geothermal Company)')

    q4 = 800
    x4 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q4, a, b, c, 0, x0)[1]
    ax1.plot(t1, x4, 'red', label='Prediction when q = 800 (Compromised Rate)')

    # Axis information
    ax1.set_title('Pressure Forecast')
    ax1.set_ylabel('Pressure (Bar)')
    ax1.set_xlabel('Time (Years)')

    ax1.legend()
    plt.show()


# This function computes uncertainty in your model
def plot_x_uncertainty():
    """
    This function plots the uncertainty of the ODE model.
    """

    #   Guess Parameters
    a = 0.001855863594537528
    b = 0.09308619728884998
    c = 0.006336076723279304
    x0 = 53.77866150176843
    pars = [a, b, c, x0]
    [t,q,x] = np.genfromtxt("TotalDataSet.csv",delimiter=',',skip_header=1).T

    # # Optimise parameters for model fit
    # pars, pars_cov = x_pars(pars_guess)

    # Store optimal values for later use

    [a, b, c, x0] = pars

    # Solve ODE and plot model
    x_e = x_curve_fitting(t, *pars)
    figa, ax1 = plt.subplots()
    ax1.plot(t, x, 'r.', label='data')
    ax1.plot(t, x_e, 'black', label='Model')

    # Remember the last time
    t_end = t[-1]

    # Create forecast time with 400 new time steps
    t1 = []
    for i in range(50):
        t1.append(i + t_end)

    # Set initial and ambient values for forecast
    xi = x_e[-1]  # Initial value of x is final value of model fit

    # Solve ODE prediction for scenario 1 (Iwi)
    q1 = 700  # heat up again
    x1 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q1, a, b, c, 0, x0)[1]
    ax1.plot(t1, x1, 'purple', label='Prediction when q = 700 (Rate desired by Iwi)')

    # Solve ODE prediction for scenario 2 (Homeowners)
    q2 = 900  # keep q the same
    x2 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q2, a, b, c, 0, x0)[1]
    ax1.plot(t1, x2, 'green', label='Prediction when q = 900 (Rate desired by Homeowners)')

    # Solve ODE prediction for scenario 3 (Geothermal Company)
    q3 = 1250  # extract at faster rate
    x3 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q3, a, b, c, 0, x0)[1]
    ax1.plot(t1, x3, 'blue', label='Prediction when q = 1250 (Rate desired by Geothermal Company)')

    # Solve ODE prediction for scenario 4 (Medium)
    q4 = 800
    x4 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q4, a, b, c, 0, x0)[1]
    ax1.plot(t1, x4, 'red', label='Prediction when q = 800 (Compromised Rate)')

    # Variance
    # var = 0.000961
    var = calculate_variance()

    # using Normal function to generate 500 random samples from a Gaussian distribution
    samples = np.random.normal(b, var, 500)

    # initialise list to count parameters for histograms
    b_list = []
    # loop to plot the different predictions with uncertainty
    for i in range(0, 499):  # 500 samples are 0 to 499
        # frequency distribution for histograms for parameters
        b_list.append(samples[i])

        spars = [a, samples[i], c, x0]
        x = x_curve_fitting(t, *spars)
        ax1.plot(t, x, 'black', alpha=0.1, lw=0.5)
        xi = x[-1]

        # Solve ODE prediction for scenario 1
        q1 = 700  # heat up again
        x1 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q1, a, samples[i], c, 0, x0)[1]
        ax1.plot(t1, x1, 'purple', alpha=0.1, lw=0.5)

        # Solve ODE prediction for scenario 2
        q2 = 900  # keep q the same
        x2 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q2, a, samples[i], c, 0, x0)[1]
        ax1.plot(t1, x2, 'green', alpha=0.1, lw=0.5)

        # Solve ODE prediction for scenario 3
        q3 = 1250  # extract at faster rate
        x3 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q3, a, samples[i], c, 0, x0)[1]
        ax1.plot(t1, x3, 'blue', alpha=0.1, lw=0.5)

        q4 = 800
        x4 = solve_ode_prediction(ode_model, t1[0], t1[-1], t1[1] - t1[0], xi, q4, a, samples[i], c, 0, x0)[1]
        ax1.plot(t1, x4, 'red', alpha=0.1, lw=0.5)

        # q5 = 400
        # x5 = solve_ode_prediction(ode_model, t1[0], t1[2], t1[1] - t1[0], xi, q5, a, samples[i], c, 0, x0)[
        #     1]
        # ax1.plot(t1[:3], x5, 'orange', alpha=0.1, lw=0.5)
        #
        # q6 = 900
        # x6 = solve_ode_prediction(ode_model, t1[2], t1[-1], t1[1] - t1[0], x5[-1], q6, a, samples[i], c, 0, x0)[1]
        # ax1.plot(t1[2:], x6, 'orange', alpha=0.1, lw=0.5)

    ax1.set_title('Pressure')
    ax1.set_ylabel('Pressure (Bar)')
    ax1.set_xlabel('Time Years')
    ax1.legend()

    # plotting the histograms
    figb, (ax2) = plt.subplots(1, 1)
    num_bins = 30
    ax2.hist(b_list, num_bins)
    ax2.set_title("Frequency Density plot for Parameter b", fontsize=9)
    ax2.set_xlabel('Parameter b', fontsize=9)
    ax2.set_ylabel('Frequency density', fontsize=9)
    a_yf5, a_yf95 = np.percentile(b_list, [5, 95])
    ax2.axvline(a_yf5, label='95% interval', color='r', linestyle='--')
    ax2.axvline(a_yf95, color='r', linestyle='--')
    ax2.legend(loc=0, fontsize=9)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def calculate_variance():
    data_points = np.random.uniform(0.1, 0.3, 100)

    # Step 1: Calculate the mean of the data points
    mean_value = sum(data_points) / len(data_points)

    # Step 2: Calculate the squared differences between each data point and the mean
    squared_diff = [(x - mean_value) ** 2 for x in data_points]

    # Step 3: Compute the sum of these squared differences
    sum_squared_diff = sum(squared_diff)

    # Step 4: Divide the sum by the total number of data points to get the variance
    variance = sum_squared_diff / len(data_points)

    print(f"The variance of the dataset is: {variance}")
    return variance

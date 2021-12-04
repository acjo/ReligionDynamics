#religions.py
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp

def define_ode_system(n, C):
    '''
    This returns a function defining the ODE system used in solve_ivp or odeint
    :param n: ( n, ) np.ndarray difference between birth rate and death rate of the ith relgion
    :param C: ( n, n ) np.ndarray matrix containing conversion rate where C[i,j]
                       represents the conversion rate from j to i (diagonal is 0)
    :return: callable the ode system
    '''

    def ode(t, P):
        '''
        :param t: independent variable
        :param P: population change as a function of time
        :return: (np.ndarray) ode system
        '''
        # system = np.zeros_like(n)
        # size = system.size
        # for i in range(size):
        # system[i] = P[i] * (n[i] + sum([(C[i,j]-C[j,i])*P[j] for j in range(size)]))
        # return system
        size = n.size
        return np.array([P[i] * (n[i] + sum([(C[i, j] - C[j, i])*P[j]
                                             for j in range(size)]))
                                             for i in range(size)])
    return ode

def numerical_solve(t_span, M, n, C, method='ivp', P0 = None, BC = None):
    '''
    Numerical solves the ivp using solve_ivp
    :parama P0: (n, ) np.ndarray array containing the initial value
    :param t_span: (tuple) contains the starting and ending time value
    :param M: ( int ) temporal discretization
    :param n: (n, ) np.ndarray difference between birth rate and death rate of the ith relgion
    :param C: (n, n) np.ndarray matrix containing conversion rate differences
    :param P0: (n, ) np.ndarray array containing the initial value, if method is 'bvp' then this is the initial guess
    :param BC: (n, ) np.ndarray boundary conditions
    :param method ="ivp": boolean determing the method
    :return: callable the ode system
    '''
    ode = define_ode_system(n, C)
    t_eval = np.linspace(t_span[0], t_span[-1], M)
    if method == 'ivp':
        sol =  solve_ivp(ode, t_span, P0, t_eval=t_eval)
    elif method == 'bvp':
        sol = solve_bvp(ode, BC, t_eval, P0 )
    else:
        raise ValueError('incorrect method')

    return sol


def plot_sol(t_span, M, n, C, method='ivp', P0=None, BC=None, labels=[]):
    '''
    plots the solution
    '''
    sol = numerical_solve(t_span, M, n, C, method=method, P0=P0, BC=BC)
    if method == 'ivp':
        fig = plt.figure()
        fig.set_dpi(300)
        ax = fig.add_subplot(111)
        for s, lab in zip(sol.y, labels):
            ax.plot(sol.t, s, label=lab)
            ax.legend(loc='best')

        ax.set_xlabel('Time')
        ax.set_ylabel('Population')
        ax.set_title('SIR Model of Religions')
        plt.show()

    elif method == 'bvp':
        fig = plt.figure()
        fig.set_dpi(300)
        ax = fig.add_subplot(111)
        for s, lab in zip(sol.y, labels):
            ax.plot(sol.x, s, label=lab)
            ax.legend(loc='best')
        ax.set_xlabel('Time')
        ax.set_ylabel('Population')
        ax.set_title('SIR Model of Religions')
        plt.show()
    else:
        raise ValueError('Incorrect method')

    return


def BryceSection():
    P0, t_span, M, n, C, method, labels = np.array([10, 10, 10]), (0, 400),20000, np.array([.2, -0.02,-0.01]), np.array([[0, .001, .001], [.01, 0,.02],[.01,.02,0]]), 'ivp', ['No Conversion', 'A', 'B']
    plot_sol(t_span, M, n, C, method, P0= P0, labels=labels)
    #plot_sol([10,10], (0,10), 1, [0,0], np.array([[0,0.1],[0,0]]), method='ivp', labels=['A','B'])
BryceSection()

# P[i] - normal people
# P[ i ] - Caelan
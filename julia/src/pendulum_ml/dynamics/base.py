from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    """ Pendulum parameters"""
    m: float
    L: float
    c: float
    g: float
    dt: float

def euler_step(x, u, f, p:Params):
    """ Simple Euler integration step.

    Args:
        x (float): current state
        u (float): current control input
        f (callable): dynamics function f(x,u,p)
        p (Params): pendulum parameters

    Returns:
        float: next state after one Euler step
    """
    return x + p.dt * f(x,u,p)

def rk4_step(x, u, f, p:Params):
    """ Runge-Kutta 4th order integration step.

    Args:
        x (float): current state
        u (float): current control input
        f (callable): dynamics function f(x,u,p)
        p (Params): pendulum parameters

    Returns:
        float: next state after one RK4 step
    """
    k1 = f(x,u,p)
    k2 = f(x + 0.5*p.dt*k1, u, p)
    k3 = f(x + 0.5*p.dt*k2, u, p)
    k4 = f(x + p.dt*k3, u, p)
    return x + (p.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

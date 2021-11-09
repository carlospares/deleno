import numpy as np

IDENTITY = 0
SUBDOMAIN = 1
POINT = 2

observables_dict = {"identity": IDENTITY, "id": IDENTITY, "subdomain": SUBDOMAIN, "point": POINT}

def observables_factory(obs_name):
    if observables_dict[obs_name]==IDENTITY:
        return identity
    elif observables_dict[obs_name]==SUBDOMAIN:
        return mass_subdomain
    elif observables_dict[obs_name]==POINT:
        return point
    else:
        raise ValueError(f"Observable {obs_name} not known to observables_factory.")


def identity(u):
    """ returns the field itself """
    return u

def mass_subdomain(u):
    """ returns the total mass in domain [1/4,1/2]x[1/4,1/2] for first component.
    WARNING: This assumes without checking that Nx and Ny are divisible by 4! """
    Nx = u.shape[-2]
    Ny = u.shape[-1]
    V = 1/(Nx*Ny)
    return np.sum(u[0, Nx//4:Nx//2, Ny//4:Ny//2])*V

def point(u):
    """ returns the value at (1/4, 1/4) for first component.
        WARNING: This assumes without checking that Nx and Ny are divisible by 4! """
    Nx = u.shape[-2]
    Ny = u.shape[-1]
    V = 1/(Nx*Ny)
    return np.sum(u[0, Nx//4, Ny//4])*V


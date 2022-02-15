import numpy as np
from numba import jit, prange
import time

# pyLJ.py - a collection of some common functions for running MD and MC simulations of a Lennard-Jones fluid

def GenerateLJLattice(size, lattice_scaling):
    """
    Generates a 3D lattice of LJ atoms, separated by Rm from each other (2^(1/6)*sigma) in reduced units
    """
    R = 2**(1/6)
    grid = np.mgrid[0:size, 0:size, 0:size]
    points = np.resize(grid[0], (size,1))
    x = np.resize(grid[0], (size**3,1))
    y = np.resize(grid[1], (size**3,1))
    z = np.resize(grid[2], (size**3,1))
    points = np.concatenate((x,y,z), axis = 1)
    lattice = points*R*lattice_scaling
    return(lattice)

@jit(nopython = True)
def DistanceComponentPBC(x1, x2, xsize):
    """
    Computes one PBC distance component
    """
    dx = x2 - x1
    dx += -xsize*int(round(dx/xsize))
    return(dx)

@jit(nopython = True)
def DistancePBC(point1, point2, box):
    """
    Computes a periodic boundary distance
    """
    SqDistance = 0.0
    for i in range(len(point1)):
        SqDistance += DistanceComponentPBC(point1[i], point2[i], box[i])**2
    Distance = SqDistance**0.5
    return(Distance)

@jit(nopython = True)
def TotalPotentialEnergy(coordinates, box):
    """
    Computes the total potential energy for a set of particles with given coordinates
    """
    total_potential_energy = 0.0
    for i in range(len(coordinates)):
        for j in range(i):            
            total_potential_energy += (4*((1/DistancePBC(coordinates[i], coordinates[j], box))**12 - 
                                          (1/DistancePBC(coordinates[i], coordinates[j], box))**6)) 
    
    return(total_potential_energy)
import sys
import numpy as np
import time
from numba import jit, prange
import matplotlib.pyplot as plt

from pyLJ import DistancePBC

header = """
##############
### rdf.py ###
##############

Author: Mikhail Ivanov
Originally written in 2019, minor revisions in 2022

RDF analysis of a Lennard-Jones fluid code 
Written as my first programming project for the 'Simulation Methods in Statistical Physics' course

All the critical expressions for computations are borrowed either from the lectures or the Rahman (1964) paper
"""

# Some unit transformations:
# Atomic mass unit to kg
amu = 1.66605304*10**(-27)
# eV to J
eV = 1.6021766208*10**(-19)
# Boltzmann constant, J/K
kB = 1.38064852*10**(-23)
# Setting the name of the file containing information for RDF calculation
RDF_file = sys.argv[1]

def RDFinit(RDF_file):
    """
    Reads a file that contains parameters for RDF calculation
    """
    data = []
    RDF_parm = []
    
    file = (r for r in open(RDF_file) if not r[0] in ('#'))
    for line in file:
        data.append(line.split()[0])
    file.close()
    
    for i in range(len(data)):
        if data[i] == 'RDF':
            RDF_parm.append(data[i+1])
            RDF_parm.append(data[i+2])
            RDF_parm.append(data[i+3])
        else:
            continue
    
    return(RDF_parm)

@jit(nopython = True)
def RDFstep(positions):
    """
    Calculates RDF for one timestep, reduced units are utilized for the computation
    Expression is borrowed from Rahman (1964) paper
    """
    N = len(positions)
    V = (box_side/sigma)**3
    dist = np.arange(dr, rmax, dr)
    total_RDF = np.zeros((N, len(dist)))
    RDF = np.zeros(dist.shape)
    
    for j in range(N):
        distances = np.zeros((N))
        for i in range(N):
            distances[i] = DistancePBC(positions[j], positions[i], box)

        for r in range(len(dist)-1):
            N_layer = 0
            for i in range(N):
                if (distances[i] > dist[r] and distances[i] <= dist[r+1]):
                    N_layer += 1
            total_RDF[j][r+1] = (V/N) * (N_layer/(4*dr*np.pi*dist[r]**2))
        RDF = np.sum(total_RDF, axis = 0)/N
    return(RDF)

@jit(nopython = True, parallel = True)
def RDFcalc(output):
    """
    Averaging RDFs over all timesteps > Teq
    """
    dist = np.arange(dr, rmax, dr)
    total_RDFstep = np.zeros((int((steps-Eq)/outfreq), len(dist)))
    RDF = np.zeros(dist.shape)
    
    trajectory = np.zeros((int((steps-Eq)/outfreq), N, 3))
    
    for t in range(len(trajectory)):
        trajectory[t] = (output[(t*N):((t+1)*N), :3])/sigma
    for t in prange(len(trajectory)):
        total_RDFstep[t] = RDFstep(trajectory[t])
    RDF = np.sum(total_RDFstep, axis = 0)/(int(steps/outfreq))
    return(dist, RDF, total_RDFstep)

@jit(nopython = True)
def CalculateCN(dist, RDF):
    """
    Calculated the coordination number
    """
    CN = np.zeros(RDF.shape)
    for i in range(len(CN)):
        CN[i+1] = CN[i] + (RDF[i+1] * 4*np.pi * dr*sigma * dist[i+1]**2 * (N)/(volume_ang))
    return(CN)

def ReadTraj(filename):
    """
    Reads trajectory file and cuts the equilibration part
    """
    output = np.loadtxt(filename)
    return(output[int((1+Eq_recorded/outfreq)*N):])

def ReadParameters(filename):
    """
    Reads simulation parameters from the dat file
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        MC = False
        MD = False
        if 'Monte Carlo' in lines[0]:
            MC = True
            print(f'Reading Monte Carlo data from {filename}...')
        elif 'Molecular Dynamics' in lines[0]:
            MD = True
            print(f'Reading Molecular Dynamics data from {filename}...')
        
        N = int(lines[1].split()[5]) # Number of particles
        atomname = lines[2].split()[2] # Atom name
        sigma = float(lines[3].split()[2]) # Sigma, A
        T = float(lines[5].split()[2]) # Temperature, K
        density = float(lines[6].split()[2]) # Density, kg/m3
        box_side = float(lines[7].split()[4]) # Box side, A
        Eq = int(lines[9].split()[5]) # N of equilibration steps
        steps = int(lines[10].split()[5]) # N of production steps
        outfreq = int(lines[11].split()[3]) # Output frequency
        dt = 0 # Set time step to 0 for MC case
        if MD:
            dt = float(lines[8].split()[3]) # Time step, ps
        
        return(MD, MC, N, atomname, sigma, T, density, box_side, Eq, steps, outfreq, dt)

def SaveRDF(dist, RDF, CN):
    """
    Saves RDF and CN data into an output file
    """
    if MD:
        filename = f'RDF_CN_{steps*dt}ps_{density}_{T}K.dat'
    elif MC:
        filename = f'RDF_CN_{steps/1E6}M_{density}_{T}K.dat'
    RDFout = np.array([dist, RDF, CN])
    if MD:
        header = f'Radial distribution function {atomname}-{atomname} for {N} particles \
at {T:.1f} K and {density:.1f} kg/m3 density \nAveraged over {int((steps-Eq)/outfreq)} time frames, \
corresponding to {(steps-Eq)*dt:.3f} ps of simulation time \nr, A; g(r); CN'
    elif MC:
        header = f'Radial distribution function {atomname}-{atomname} for {N} particles \
at {T:.1f} K and {density:.1f} kg/m3 density \nAveraged over {int((steps-Eq)/outfreq)} time frames \n\
Total number of MC steps is {steps} \nr, A; g(r); CN'
    np.savetxt(filename, RDFout.T, fmt='%-7.4f', header=header)
    return

def plotRDF():
    """
    Plots the resulting RDF
    """
    RDF_max = np.max(RDF)
    if MD:
        RDFplotname = 'RDF-MD.pdf'
    elif MC:
        RDFplotname = 'RDF-MC.pdf'
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(12,7))
    plt.plot(dist, RDF, color='navy', lw=3, label=(rf'RDF, $\rho$ = {density:.0f} kg/$m^3$, T = {T:.0f} K'))
    plt.legend(loc="upper right")
    plt.ylabel('g(r)')
    plt.xlabel('r, Å')
    plt.xlim(2, rmax*sigma)
    plt.ylim(0, RDF_max+0.5)
    plt.grid()
    plt.savefig(RDFplotname)
    return

if __name__ == "__main__":
    print(header)
    # Get RDF data
    RDF_parm = RDFinit(RDF_file)
    # Trajectory filename
    trajname = RDF_parm[2]
    # Read parameters
    MD, MC, N, atomname, sigma, T, density, box_side, Eq, steps, outfreq, dt = ReadParameters(trajname)
    Eq_recorded = int(Eq/outfreq)
    volume_ang = box_side**3  # volume of the system in Angstrom^3
    box = np.array([box_side/sigma, box_side/sigma, box_side/sigma])
    # RDF parameters
    dr = float(RDF_parm[0]) / sigma
    rmax = float(RDF_parm[1]) / sigma
    # Read trajectory
    output = ReadTraj(trajname)
    # Calculate RDF and CN
    start = time.time()                  
    RDF_data = RDFcalc(output)
    end = time.time()
    dist = RDF_data[0]*sigma
    RDF = RDF_data[1]
    totalRDF = RDF_data[2]
    CN = CalculateCN(dist, RDF)
    print('RDF calculation time = ' + format(end-start, '#.3f') + ' seconds')
    # Save RDF data
    SaveRDF(dist, RDF, CN)
    # Plot RDF data
    plotRDF()

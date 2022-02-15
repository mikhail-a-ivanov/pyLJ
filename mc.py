import sys
import numpy as np
from numpy import random as r
import time
from numba import jit
from pyLJ import GenerateLJLattice, DistancePBC, TotalPotentialEnergy

header = """
#############
### mc.py ###
#############

Author: Mikhail Ivanov
Originally written in 2019, minor revisions in 2022

Monte Carlo simulation of a Lennard-Jones fluid code 
Written as my first programming project for the 'Simulation Methods in Statistical Physics' course

Warning! The code does its job, but is extremely inefficient and naive
All the critical expressions for computations are borrowed either from the lectures or the Rahman (1964) paper
"""

# Some unit transformations:
# Atomic mass unit to kg
amu = 1.66605304*10**(-27)
# eV to J
eV = 1.6021766208*10**(-19)
# Boltzmann constant, J/K
kB = 1.38064852*10**(-23)
# Setting the name of the file containing forcefield information and MC parameters (number of steps, displacement step length)
MC_file = sys.argv[1]

def MCinit(MC_file):
    """
    Reads the MCinit file, extracts the forcefield and MC paramters
    forcefield - MCinit[0], MCparameters - MCinit[1], size = MCinit[2]
    """
    data = []
    forcefield = []
    MC_parameters = []
    lattice_data = []
    
    file = (r for r in open(MC_file) if not r[0] in ('#'))
    for line in file:
        data.append(line.split()[0])
    file.close()
    
    for i in range(len(data)):
        if data[i] == 'Forcefield':
            forcefield.append(data[i+1])
            forcefield.append(data[i+2])
            forcefield.append(data[i+3])
            forcefield.append(data[i+4])
        if data [i] == 'MCparameters':
            MC_parameters.append(data[i+1])
            MC_parameters.append(data[i+2])
            MC_parameters.append(data[i+3])
            MC_parameters.append(data[i+4])
            MC_parameters.append(data[i+5])
        if data [i] == 'Lattice':
            lattice_data.append(data[i+1])
            lattice_data.append(data[i+2])
        else:
            continue
    
    return(forcefield, MC_parameters, lattice_data)

@jit(nopython = True)
def ParticleEnergy(coordinates, particleNum):
    """
    Calculates the energy of one particle, surrounded by the others
    """
    particle = coordinates[particleNum]
    particle_energy = 0.0
    for j in range(len(coordinates)):
        if np.all(np.equal(particle, coordinates[j])):
            continue
        else:
            particle_energy += (4*((1/DistancePBC(particle, coordinates[j], box))**12 - 
                                   (1/DistancePBC(particle, coordinates[j], box))**6)) 
    return(particle_energy)

@jit(nopython = True)
def MCmove(E, coordinates):
    """
    Monte Carlo move: Picks a particle and calculates its energy
    """
    particleNum = r.randint(0, len(coordinates))
    oldE = ParticleEnergy(coordinates, particleNum)
    
    # Displaces the particle and calculates the new energy
    displacement = np.array([delta*(r.ranf() - 0.5), delta*(r.ranf() - 0.5), delta*(r.ranf() - 0.5)])
    coordinates[particleNum] += displacement
    newE = ParticleEnergy(coordinates, particleNum)
    
    # Calculates the energy difference
    deltaE = newE - oldE
    
    # Acceptance and rejections counters
    accepted = 0
    rejected = 0
    
    # Accepts or rejects the move
    if deltaE < 0:
        accepted += 1
        #Eout = E + deltaE
        E += deltaE
    else:
        if r.ranf() < np.exp(-(1/T_red) * deltaE):
            accepted += 1
            #Eout = E + deltaE
            E += deltaE
        else:
            rejected += 1
            coordinates[particleNum] -= displacement
            
    return(coordinates, E, accepted, rejected)

@jit(nopython = True)
def MCequilibration(coordinates):
    """
    Performs Monte Carlo steps to equilibrate the system
    """
    E = TotalPotentialEnergy(coordinates, box)
    energies = np.empty(Eq+1)
    energies[0] = E
    accepted = 0
    rejected = 0
    
    for i in range(Eq):
        MCout = MCmove(energies[i], coordinates)
        energies[i+1] = MCout[1]
        accepted += MCout[2]
        rejected += MCout[3]
        
    coordinates = MCout[0]
    ratio = accepted / (accepted + rejected)
    
    return(coordinates, energies[::outfreq], ratio)

def SaveMCData(data, convert, filename, header):
    """
    Saves data (energies, positions) from MC simulations
    """
    np.savetxt(filename, data.T*convert, fmt='%-7.4f', header = header)
    return

def SaveXYZConf(filename):
    """
    Saves the equilibrated configuration in VMD .xyz format
    """
    coordinates = MCEqData[0] * sigma
    xyz = []
    for j in range(len(lattice)):
        xyz.append([])
        xyz[j].append(atomname)
    for j in range(len(lattice)):
        for i in range(3):
            if coordinates[j][i] >= box[i]*sigma:
                coordinates[j][i] += -box[i]*sigma
            if coordinates[j][i] < 0.0:
                coordinates[j][i] += box[i]*sigma
            xyz[j].append(format(coordinates[j][i], '#.7g').rjust(10))
        xyz[j].append('  ')
    
    output_file = open(filename, 'w+')
    output_file.write(str(len(xyz)) + '\n')
    output_file.write('Step = 0\n')
    for j in range(len(xyz)):
        for i in range(len(xyz[j])):
            output_file.write(str(xyz[j][i]) + ' ')
        output_file.write('\n')
    
    output_file.close()
    return

@jit(nopython = True)
def MC(coordinates):
    """
    Performs Monte Carlo steps to sample the system's configurational space
    """
    E = TotalPotentialEnergy(coordinates, box)
    energies = np.empty(steps+1)
    energies[0] = E
    accepted = 0
    rejected = 0
    MCtraj = np.empty((int(steps/outfreq), lattice.shape[0], lattice.shape[1]))
    
    for i in range(steps):
        MCout = MCmove(energies[i], coordinates)
        energies[i+1] = MCout[1]
        accepted += MCout[2]
        rejected += MCout[3]
        if i % outfreq == 0:
            MCtraj[int(i / outfreq)] = MCout[0]
    
    ratio = accepted / (accepted + rejected)

    return(MCtraj, energies[::outfreq], ratio)

def WriteData(data, filename):
    """
    Saves MC trajectory in unwrapped format
    """
    with open(filename, 'w') as output_file:
        output_file.write(f'# Monte Carlo run \n')
        output_file.write(f'# Number of LJ particles: {len(lattice)} \n')
        output_file.write(f'# Atomname: {atomname} \n')
        output_file.write(f'# Sigma: {sigma:.3f} A \n')
        output_file.write(f'# Epsilon: {eps:.3e} J \n')
        output_file.write(f'# Temparature: {T:.1f} K \n')
        output_file.write(f'# Density: {density:.1f} kg/m3 \n')
        output_file.write(f'# Cubic box size: {box[0]*sigma:.3f} A \n')
        output_file.write(f'# Max displacement step: {delta:.2f} A \n')
        output_file.write(f'# Number of equilibration steps: {Eq} \n')
        output_file.write(f'# Number of production steps: {steps} \n')
        output_file.write(f'# Output frequency: {outfreq} \n')
        
        for i in range(len(data)):
            output_file.write('# Step = ' + str(int(i*outfreq)) + '\n')  
            np.savetxt(output_file, data[i], fmt='%-7.4f')
            
        output_file.write('# Production simulation time: ' + format(MCTime, '#.3f') + ' seconds')
        output_file.close()
    return

def SaveXYZTrajectory(data, filename):
    """
    Saves MC trajectory in wrapped .xyz VMD format
    """
    xyz = []
    for t in range(len(data)):
        xyz.append([])
        for j in range(len(lattice)):
            xyz[t].append([])
            xyz[t][j].append(atomname)
    for t in range(len(data)):
        for j in range(len(lattice)):
            for i in range(3):
                if data[t][j][i] >= box[i]*sigma:
                    data[t][j][i] += -box[i]*sigma
                if data[t][j][i] < 0.0:
                    data[t][j][i] += box[i]*sigma
                xyz[t][j].append(format(data[t][j][i], '#.7g').rjust(10))
            xyz[t][j].append('  ')
    
    output_file = open(filename, 'w+')
    for t in range(len(xyz)):
        output_file.write(str(len(xyz[t])) + '\n')
        output_file.write('Step = ' + str(t*outfreq) + '\n')
        for j in range(len(xyz[t])):
            for i in range(len(xyz[t][j])):
                output_file.write(str(xyz[t][j][i]) + ' ')
            output_file.write('\n')
    
    output_file.close()
    return

if __name__ == "__main__":
    print(header)
    # Get MC data
    MCinit_data = MCinit(MC_file)
    forcefield = MCinit_data[0]
    MC_parameters = MCinit_data[1]
    lattice_data = MCinit_data[2]
    # Some global arguments:
    # Forcefield parameters
    atomname = forcefield[0] # name of the atom
    atommass = float(forcefield[1]) # amu
    eps = float(forcefield[2]) * (10**-21) # Joules
    sigma = float(forcefield[3]) # Å
    # Energy conversion from J to eV
    convert_E = eps/eV
    # MC parameters
    steps = int(MC_parameters[0]) # N steps
    delta = float(MC_parameters[1]) # Length of the displacement step, Å
    outfreq = int(MC_parameters[2]) # Output frequency
    T = float(MC_parameters[3]) # Temperature, K
    T_red = kB*T/eps # Temperature in reduced units
    Eq = int(MC_parameters[4]) # Number of equilibration steps
    # Lattice 
    size = int(lattice_data[0]) # Number of lattice points
    density = float(lattice_data[1]) # Desired density of the system, kg/m^3
    density_Rm = (amu*atommass / (2**(1/6) * sigma * 10**(-10))**3) # Density of the LJ lattice with Rm distance
                                                                    # between the neighbouring atoms
    lattice_scaling = (density_Rm / density)**(1/3) # distance scaling factor for achieving the desired denstiy
    box = np.array([size*(2**(1/6))*lattice_scaling, size*(2**(1/6))*lattice_scaling, 
                    size*(2**(1/6))*lattice_scaling]) # Simulation box vectors in reduced units
    # Generate lattice
    lattice = GenerateLJLattice(size, lattice_scaling)
    # Calculate the density of the system in kg/m3
    mass = len(lattice) * atommass * amu # mass of the system in kg
    volume = (box[0] * sigma * 10**(-10))**3 # volume of the system in m^3
    density_calc = mass / volume # density in kg/m^3
    
    print(f'Running MC simulation of {int(size**3)} LJ particles at {T:.1f} K...\n')
    print(f'Number of equilibration MC steps: {Eq}')
    print(f'Number of production MC steps: {steps}')
    print(f'Maximum displacement step along each axis: {delta:.2f} Angstrom')
    print(f'Cubic box side: {box[0]*sigma:.3f} A')
    print(f'Density: {density:.3f} kg/m^3')
    print(f'Output frequency: {outfreq}\n')

    # Start the equilibration and measure simulation time 
    print('Running Monte Carlo equilibration...')
    start = time.time()
    MCEqData = MCequilibration(lattice)
    end = time.time()
    MCEqTime = end - start
    print('Equilibration time: ' + format(MCEqTime, '#.3f') + ' seconds')
    print('Acceptance ratio = ' + str(MCEqData[2]))
    
    # Start the production simulation and measure simulation time 
    print('\nRunning Monte Carlo production simulation...')
    start = time.time()
    MCData = MC(MCEqData[0])
    end = time.time()
    MCTime = end - start
    print('Production simulation time: ' + format(MCTime, '#.3f') + ' seconds')
    print('Acceptance ratio = ' + str(MCData[2]))
    
    # Some headers for output files
    header_energy_eq = ('Monte Carlo equilibration energies (eV) for the system of ' + 
                        str(len(lattice)) + ' ' + str(atomname) +
                    ' atoms at ' + str(T) + ' K and ' + str(round(density)) + ' kg/m^3 density\n' +
                    'Number of performed MC equilibration steps = ' + str(Eq) + '\n' + '# Output frequency is: ' + str(outfreq) + '\n' 
                    + 'Displacement step = ' + str(delta) + ' Angstrom\n' + 'Acceptance ratio = ' + str(MCEqData[2]) + '\n' + 
                    'Calculation time: ' + format(MCEqTime, '#.3f') + ' seconds')
    
    
    header_energy = ('Monte Carlo energies (eV) for the system of ' + str(len(lattice)) + ' ' + str(atomname) +
                    ' atoms at ' + str(T) + ' K and ' + str(round(density)) + ' kg/m^3 density\n' +
                    'Number of performed MC steps = ' + str(steps) + '\n' + '# Output frequency is: ' + str(outfreq) + '\n'
                    + 'Displacement step = ' + str(delta) + ' Angstrom\n' + 'Acceptance ratio = ' + str(MCData[2]) + '\n' + 
                    'Calculation time: ' + format(MCTime, '#.3f') + ' seconds')
    
    # Save some data to the output files
    energies_eq_name = f'energies_{Eq/1E6:.1f}M_eq_{density:.1f}_{T:.1f}K.dat'
    energies_name = f'energies_{steps/1E6:.1f}M_prod_{density:.1f}_{T:.1f}K.dat'
    confout_name = f'confout_{Eq/1E6:.1f}M_eq_{density:.1f}_{T:.1f}K.xyz'
    MCdata_name = f'MCtraj_{steps/1E6:.1f}M_{density:.1f}_{T:.1f}K.dat'
    MCxyz_name = f'MCtraj_{steps/1E6:.1f}M_{density:.1f}_{T:.1f}K.xyz'

    SaveMCData(MCEqData[1], convert_E, energies_eq_name, header_energy_eq)
    SaveMCData(MCData[1], convert_E, energies_name, header_energy)
    SaveXYZConf(confout_name)
    WriteData(MCData[0]*sigma, MCdata_name)
    SaveXYZTrajectory(MCData[0]*sigma, MCxyz_name)

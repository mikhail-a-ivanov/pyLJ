import sys
import numpy as np
from numba import jit, prange
import time

from pyLJ import GenerateLJLattice, DistanceComponentPBC, DistancePBC, TotalPotentialEnergy

header = """
#############
### md.py ###
#############

Author: Mikhail Ivanov
Originally written in 2019, minor revisions in 2022

Molecular dynamics of a Lennard-Jones fluid code 
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
# Setting the name of the file containing forcefield information and MD parameters (number of steps, step length)
MD_file = sys.argv[1]

def MDinit(MD_file):
    """
    Reads the MDinit file, extracts the forcefield and MD paramters
    forcefield - MDinit[0], MDparameters - MDinit[1], size = MDinit[2]
    """
    data = []
    forcefield = []
    MD_parameters = []
    lattice_data = []
    
    file = (r for r in open(MD_file) if not r[0] in ('#'))
    for line in file:
        data.append(line.split()[0])
    file.close()
    
    for i in range(len(data)):
        if data[i] == 'Forcefield':
            forcefield.append(data[i+1])
            forcefield.append(data[i+2])
            forcefield.append(data[i+3])
            forcefield.append(data[i+4])
        if data[i] == 'MDparameters':
            MD_parameters.append(data[i+1])
            MD_parameters.append(data[i+2])
            MD_parameters.append(data[i+3])
            MD_parameters.append(data[i+4])
            MD_parameters.append(data[i+5])
        if data[i] == 'Lattice':
            lattice_data.append(data[i+1])
            lattice_data.append(data[i+2])
        else:
            continue
    
    return(forcefield, MD_parameters, lattice_data)

def GenerateVelocities():
    """
    Generates velocities in reduced units for a 3D lattice of atoms
    according to Maxwell-Boltzmann distribution.
    Center-of-mass motion is removed, velocities are scaled to the input temperature
    """
    # Velocity generation according to the Maxwell-Boltzmann distribution
    initial_velocities = np.empty((np.ndarray.transpose(lattice)).shape)
    for i in range(len(initial_velocities)):
        initial_velocities[i] = np.random.normal(loc = 0.0, scale = vi_STD, 
                                                 size = (initial_velocities[i].shape))
    initial_velocities = np.ndarray.transpose(initial_velocities)
    # Removal of the center-of-mass motion
    sumV = np.sum(initial_velocities, axis=0)
    velocities = initial_velocities - sumV/len(lattice)
    # Temperature calculation
    velocities = np.ndarray.transpose(velocities)
    SqVx = np.sum(velocities[0]**2)
    SqVy = np.sum(velocities[1]**2)
    SqVz = np.sum(velocities[2]**2)
    SqMeanV = (SqVx + SqVy + SqVz) * 1 / len(lattice)
    Tcalc = atommass * amu * SqMeanV / (3*kB)
    # Temperature scaling
    velocities = np.ndarray.transpose(velocities)
    velocities = velocities*(T/Tcalc)**0.5
    # Translation to reduced units
    initial_velocities_reduced = velocities * (atommass * amu / eps)**0.5
    return(initial_velocities_reduced)

@jit(nopython = True)
def ComputeTemperature(velocities_reduced):
    """
    Takes an array of velocities in reduced units, converts them to SI and
    computes system's temperature [K]
    """
    velocities = velocities_reduced * (eps/(atommass*amu))**0.5
    SqMeanV = np.sum(np.sum(velocities**2, axis = 0))  * 1 / len(lattice)
    T = atommass * amu * SqMeanV / (3*kB)
    return(T)

@jit(nopython = True)
def TotalKineticEnergy(velocities):
    """
    Calculates the total kinetic for a set of particles with given velocities
    """     
    total_kinetic_energy = 0
    for i in range(len(velocities)):
        total_kinetic_energy += ParticleKineticEnergy(velocities[i])
    
    return(total_kinetic_energy)

@jit(nopython = True)
def ParticleKineticEnergy(velocity):
    """
    Calculates the kinetic energy of a single particle with given velocities
    """
    kinetic_energy = 0
    for i in range(len(velocity)):
        kinetic_energy += (velocity[i]**2)/2
     
    return(kinetic_energy)      

@jit(nopython = True)
def ForceCalculation(particle, coordinates):
    """
    Calculates the force acting on a single particle with given coordinates (particle argument)
    surrounded by other particles with given coordinates (coordinates argument).
    The expression is borrowed from Rahman (1964) paper     
    """     
    force = np.zeros(particle.shape)
    # Check if the particles are not the same:
    for j in range(len(coordinates)):
        if np.all(np.equal(particle, coordinates[j])):
            continue
        else:
            for i in range(len(particle)):
                force[i] += 24 * (((DistanceComponentPBC(coordinates[j][i], particle[i], box[i])) / 
                                 (DistancePBC(coordinates[j], particle, box))**2) * 
                                 (2 * (1/DistancePBC(coordinates[j], particle, box))**12 -
                                 (1/DistancePBC(coordinates[j], particle, box))**6))
            
    return(force)

@jit(nopython = True, parallel = True)
def TotalForceCalculation(coordinates):
    """
    Calculate force array for a set of particles with given coordinates (coordinates argument)
    """
    forces = np.zeros(coordinates.shape)
    for i in prange(len(coordinates)):
        forces[i] = ForceCalculation(coordinates[i], coordinates)
    return(forces)

@jit(nopython = True)
def VelocityVerletMD():
    """
    Main integrator for the equations of motions of the particles with the coordinates and velocities
    """
    trajectory = np.zeros((steps+1, lattice.shape[0], lattice.shape[1]))
    velocities = np.zeros((steps+1, lattice.shape[0], lattice.shape[1]))
    forces = np.zeros((steps+1, lattice.shape[0], lattice.shape[1]))
    
    trajectory[0] = lattice
    velocities[0] = initial_velocities
    forces[0] = TotalForceCalculation(lattice)
        
    for t in range(steps):
        trajectory[t+1] = (trajectory[t] + velocities[t]*dt + 0.5*dt**2 * forces[t])

        forces[t+1] = TotalForceCalculation(trajectory[t+1])

        velocities[t+1] = (velocities[t] + dt*(forces[t+1] + forces[t])/2)
        if t < Teq:
            velocities[t+1] = velocities[t+1] * (T/ComputeTemperature(velocities[t+1]))**0.5
        else:
            continue
    
    return(trajectory, velocities, forces)      

def SaveData():
    """
    Saves the coordinates, velocities and forces of each particle for each 'trajout'th step
    """
    data = np.zeros((int(steps/trajout)+1, lattice.shape[0], 3*lattice.shape[1]))
    data[0] = np.concatenate((trajectory[0]*sigma, velocities[0]*convert_V, 
                                      forces[0]*convert_F), axis=1)
    for t in range(steps+1):
        if t % trajout == 0:
            data[int(t/trajout)] = np.concatenate((trajectory[t]*sigma, velocities[t]*convert_V, 
                                      forces[t]*convert_F), axis=1)
    
    return(data)

def WriteData(filename):
    """
    Saves the compressed data to the output file
    """
    data = SaveData()
    with open(filename, 'w') as output_file:
        output_file.write('# The system consisting of ' + str(len(lattice))
                      + ' particles was simulated for ' + str(steps) + ' time steps'  
                      + ' with a ' + format(10**(12) * dt*sigma*10**(-10)*
                                            (atommass*amu/eps)**0.5, '#.3f') + 
                      ' ps time step length' + '\n')
        output_file.write('# Output frequency is: ' + str(trajout) + '\n')
        output_file.write('# Number of equilibration steps is: ' + str(Teq) + '\n')
        output_file.write('# Starting temperature and density are: ' + str(T) + ' K, '
                      + str(density) + ' kg/m^3' + '\n')
        output_file.write('# Simulation box: ' + str(box*sigma) + ' Angstrom\n')
        output_file.write('# X, Y, Z, [Angstrom];  Vx, Vy, Vz, [Angstrom/ps];  Fx, Fy, Fz, [eV/Angstrom]'  + '\n')
        
        for i in range(len(data)):
            output_file.write('# Step = ' + str(i*trajout) + '\n')  
            np.savetxt(output_file, data[i], fmt='%-7.4f')
            
        output_file.write('\n# Calculation time: ' + format(MD_time, '#.3f') + ' seconds')
        output_file.close()
    return

def WriteData(filename):
    """
    Saves the compressed data to the output file
    """
    data = SaveData()
    with open(filename, 'w') as output_file:
        output_file.write(f'# Molecular Dynamics run \n')
        output_file.write(f'# Number of LJ particles: {len(lattice)} \n')
        output_file.write(f'# Atomname: {atomname} \n')
        output_file.write(f'# Sigma: {sigma:.3f} A \n')
        output_file.write(f'# Epsilon: {eps:.3e} J \n')
        output_file.write(f'# Temparature: {T:.1f} K \n')
        output_file.write(f'# Density: {density:.1f} kg/m3 \n')
        output_file.write(f'# Cubic box size: {box[0]*sigma:.3f} A \n')
        output_file.write(f'# Time step: {10**(12) * dt*sigma*10**(-10)*(atommass*amu/eps)**0.5:.3f} ps \n')
        output_file.write(f'# Number of equilibration steps: {Teq} \n')
        output_file.write(f'# Number of production steps: {steps} \n')
        output_file.write(f'# Output frequency: {trajout} \n')
        
        for i in range(len(data)):
            output_file.write('# Step = ' + str(i*trajout) + '\n')  
            np.savetxt(output_file, data[i], fmt='%-7.4f')
            
        output_file.write('\n# Calculation time: ' + format(MD_time, '#.3f') + ' seconds')
        output_file.close()
    return

def GlobalDataCollection():
    """
    Collects the system properties according to the input
    """
    global_data = [[]]
    for t in range(steps):
        if t % trajout == 0:
            global_data.append([])
    
    for t in range(steps+1):
        if t % trajout == 0:
            global_data[int(t/trajout)].append(format(TotalKineticEnergy
                                       (velocities[t]) * convert_E, '#.7g').ljust(10))
    for t in range(steps+1):
        if t % trajout == 0:
            global_data[int(t/trajout)].append(format(TotalPotentialEnergy
                                       (trajectory[t], box) * convert_E, '#.7g').ljust(10))
    for t in range(steps+1):
        if t % trajout == 0:
            global_data[int(t/trajout)].append(format(TotalPotentialEnergy
                                       (trajectory[t], box) * convert_E + TotalKineticEnergy
                                       (velocities[t]) * convert_E, '#.7g').ljust(10))
    for t in range(steps+1):
        if t % trajout == 0:
                    global_data[int(t/trajout)].append(format(ComputeTemperature
                                       (velocities[t]), '#.7g').ljust(10))
        
    return(global_data)

def SaveGlobalData(filename):
    """
    Saves the selected system properties into the output file
    """
    start = time.time()
    data = GlobalDataCollection()
    output_file = open(filename, 'w+')
    output_file.write('# The system consisting of ' + str(len(lattice))
                      + ' particles was simulated for ' + str(steps) + ' time steps'  
                      + ' with a ' + format(10**(12) * dt*sigma*10**(-10)*
                                            (atommass*amu/eps)**0.5, '#.3f') + 
                      ' ps time step length' + '\n')
    output_file.write('# Output frequency is: ' + str(trajout) + '\n')
    output_file.write('# Number of equilibration steps is: ' + str(Teq) + '\n')
    output_file.write('# Starting temperature and density are: ' + str(T) + ' K, '
                      + str(density) + ' kg/m^3' + '\n')

    output_file.write('# Total Kinetic Energy, eV;')
    output_file.write(' Total Potential Energy, eV;')
    output_file.write(' Total Energy, eV;')
    output_file.write(' Temperature, K;')

    output_file.write('\n')
    for t in range(len(data)):
        output_file.write('\n'+ '# Step = ' + str(t*trajout) + '\n')
        for j in range(len(data[t])):
            output_file.write(str(data[t][j]) + ' ')
        output_file.write('\n')
    end = time.time()
    GlobalDataTime = end-start
    print('Save global data time = ' + format(GlobalDataTime, '#.3f') + ' seconds')
    output_file.write('# Save global data time = ' + format(GlobalDataTime, '#.3f') + ' seconds')
    output_file.close()
    return

def SaveXYZTrajectory(filename):
    """
    Saves the trajectory in VMD .xyz format
    """
    data = SaveData()
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
        output_file.write('Step = ' + str(t*trajout) + '\n')
        for j in range(len(xyz[t])):
            for i in range(len(xyz[t][j])):
                output_file.write(str(xyz[t][j][i]) + ' ')
            output_file.write('\n')
    
    output_file.close()
    return

if __name__ == "__main__":
    print(header)
    # Get MD data
    MDinit_data = MDinit(MD_file)
    forcefield = MDinit_data[0]
    MD_parameters = MDinit_data[1]
    lattice_data = MDinit_data[2]
    # Get forcefield parameters
    atomname = forcefield[0] # name of the atom
    atommass = float(forcefield[1]) # amu
    eps = float(forcefield[2]) * (10**-21) # Joules
    sigma = float(forcefield[3]) # Å
    # Get MD parameters
    steps = int(MD_parameters[0]) # N steps
    dt = round((float(MD_parameters[1]) * 10**(-12) * 
                (eps*(10**20)/(atommass*amu*sigma**2))**0.5),7) # time step in reduced units,
                                                                # rounded to 7th digit
    trajout = int(MD_parameters[2]) # Output frequency
    T = float(MD_parameters[3]) # Temperature, K
    T_red = kB*T/eps # Temperature in reduced units
    Teq = int(MD_parameters[4]) # Number of temperature equilibration steps
    # Define the simulation box (cubic lattice)
    size = int(lattice_data[0]) # Number of lattice points
    density = float(lattice_data[1]) # Desired density of the system, kg/m^3
    density_Rm = (amu*atommass / (2**(1/6) * sigma * 10**(-10))**3) # Density of the LJ lattice with Rm distance
                                                                    # between the neighbouring atoms
    lattice_scaling = (density_Rm / density)**(1/3) # distance scaling factor for achieving the desired denstiy
    box = np.array([size*(2**(1/6))*lattice_scaling, size*(2**(1/6))*lattice_scaling, 
                    size*(2**(1/6))*lattice_scaling]) # Simulation box vectors in reduced units
    # Conversion factors for velocities, forces and energies
    convert_V = (1/100)*(eps/(atommass*amu))**0.5
    convert_F = eps/(sigma*eV)
    convert_F_pres = eps/(sigma*10**(-10))
    convert_E = eps/eV
    vi_STD = (kB * T/(atommass * amu))**0.5 # velocity component standard deviation 
                                            # for the velocity distribution function
    # Generate lattice
    lattice = GenerateLJLattice(size, lattice_scaling)
    # Generate velocities
    initial_velocities = GenerateVelocities()
    # Calculate the density of the system in kg/m3
    mass = len(lattice) * atommass * amu # mass of the system in kg
    volume = (box[0] * sigma * 10**(-10))**3 # volume of the system in m^3
    density_calc = mass / volume # density in kg/m^3
    print(f'Running MD simulation of {int(size**3)} LJ particles at {T:.1f} K for {steps*float(MD_parameters[1])} ps...\n')
    print(f'Total number of steps: {steps}')
    print(f'Number of equilibration steps: {Teq}')
    print(f'Cubic box side: {box[0]*sigma:.3f} A')
    print(f'Density: {density:.3f} kg/m^3')
    print(f'Output frequency: {trajout}\n')
    
    # Run MD simulation
    start = time.time()
    result = VelocityVerletMD()
    end = time.time()
    trajectory = result[0]
    velocities = result[1]
    forces = result[2]
    MD_time = end-start
    print('Calculation time: ' + format(MD_time, '#.3f') + ' seconds')

    # Save data
    start = time.time()
    WriteData('traj_' + str(steps*float(MD_parameters[1])) + 'ps_' + str(density) + '_' + str(T) + 'K.dat')
    end = time.time()
    print('Write data time = ' + format(end-start, '#.3f') + ' seconds')
    SaveGlobalData('energy_' + str(steps*float(MD_parameters[1])) + 'ps_' + str(density) + '_' + str(T) + 'K.dat')
    start = time.time()
    SaveXYZTrajectory('traj_' + str(steps*float(MD_parameters[1])) + 'ps_' + str(density) + '_' + str(T) + 'K.xyz')
    end = time.time()
    print('Save XYZ time = ' + format(end-start, '#.3f') + ' seconds')

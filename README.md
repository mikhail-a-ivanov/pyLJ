# pyLJ
***A simple Python-based Molecular Dynamics and Metropolis Monte Carlo code for simulating a Lennard-Jones fluid***

Author: Mikhail Ivanov

Originally written in 2019, minor revisions in 2022

Written as my first programming project for the 'Simulation Methods in Statistical Physics' course

Warning! The code does its job, but is extremely inefficient and naive

All the critical expressions for computations are borrowed either from the lectures or the Rahman (1964) paper

# How to run
## MD:
1. Check the input settings in the `md-init.dat`
2. Run the simulation: `python md.py md-init.dat > md.log`

## MC:
1. Check the input settings in the `mc-init.dat`
2. Run the simulation: `python mc.py mc-init.dat > mc.log`

## RDF calculation:
1. Check the input settings in the `rdf-init.dat`
2. Run the RDF calculation: `python rdf.py rdf-init.dat > rdf.log`

# Required python libraries:
- `numpy`
- `numba`
- `matplotlib`

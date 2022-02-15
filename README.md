# pyLJ
***A simple Python-based Molecular Dynamics and Metropolis Monte Carlo code for simulating a Lennard-Jones fluid***

Written as my first programming project for the 'Simulation Methods in Statistical Physics' course

All the critical expressions for computations are borrowed either from the lectures or the Rahman (1964) paper

Author: Mikhail Ivanov

Originally written in 2019, minor revisions in 2022

Special thanks to Jonatan Öström (my teaching assistant from the course)

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

# Supplementary data:
- MD and MC simulation data
- Progress reports for the course
- Movies!

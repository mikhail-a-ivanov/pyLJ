# pyLJ
***A simple Python-based Molecular Dynamics and Metropolis Monte Carlo code for simulating a Lennard-Jones fluid***

Written as my first programming project for the 'Simulation Methods in Statistical Physics' course

Author: Mikhail Ivanov

Special thanks to Jonatan Öström (my teaching assistant from the course)

Originally written in 2019, minor revisions in 2022

All molecular images are generated using **VMD<sup>3**.

**References:**
1. Frenkel, Daan, and Berend Smit. "Understanding molecular simulation: from algorithms to applications." *Vol. 1. Elsevier, 2001.*
2. Rahman, Aneesur. "Correlations in the motion of atoms in liquid argon." *Physical review 136.2A (1964): A405.*
3. Humphrey, William, Andrew Dalke, and Klaus Schulten. *"VMD: visual molecular dynamics." Journal of molecular graphics 14.1 (1996): 33-38.*
4. Lectures from the course

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

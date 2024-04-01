#!/usr/bin/env python
# coding: utf-8




# The MD simulation without any reactions, just the particles moving around with temperature
# 2024.03.15
# HongKee Yoon

# pip install numpy
# pip install ase 
# pip install matplotlib




# %%
# Input
temperature_K = 300.0
supercell_size = (4,4,4)
num_steps = 500
print("Running MD simulation at", temperature_K, "K")





import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units

from ase.visualize.plot import plot_atoms

# Set up a cluster of argon atoms
atoms = Atoms('Ar', positions=[[5, 5, 5]], cell=(10, 10, 10))

plot_atoms(atoms, radii=0.3, rotation=('20x,25y,20z'))





from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.io.trajectory import Trajectory

# Set up a simple cubic lattice of argon atoms
atoms = bulk('Ar', 'sc', a=5.26, cubic=True) * supercell_size # Create a 4x4x4 supercell

plot_atoms(atoms, radii=0.3, rotation=('20x,25y,20z'))
plt.savefig('argon_crystal.png')





plt.close('all')





import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit






# Define Lennard-Jones parameters
epsilon = 0.0103  # Potential depth in eV
sigma = 3.40      # Finite distance at which the inter-particle potential is zero in Ångström

# Use Lennard-Jones potential calculator
lj_calc = LennardJones(epsilon=epsilon, sigma=sigma)
atoms.set_calculator(lj_calc)

# Set the initial temperature to 300K
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

# Remove linear momentum to avoid drifting
atoms.set_momenta(atoms.get_momenta() - atoms.get_momenta().mean(axis=0))

# Define the dynamics
dyn = VelocityVerlet(atoms, 2.5 * units.fs)  # Time step is 2.5 fs

# Setup the trajectory writer
trajectory_filename = 'argon_trajectory.traj'
traj = Trajectory(trajectory_filename, 'w', atoms)

# Attach the trajectory to the dynamics
dyn.attach(traj.write, interval=1)

# Lists to store energies
potential_energies = []
kinetic_energies = []

# Function to record the potential, kinetic and total energy
def record_energy(a=atoms):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    potential_energies.append(epot)
    kinetic_energies.append(ekin)

# Now perform the simulation and record energies
for step in tqdm(range(num_steps)):  # Number of steps
    dyn.run(100)  # Run simulation for 100 * 2.5 fs = 250 fs
    record_energy()

traj.close()

# After simulation, plot the histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(potential_energies, bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Epot')
plt.xlabel('Potential Energy (eV)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(kinetic_energies, bins=20, color='red', alpha=0.7)
plt.title('Histogram of Ekin')
plt.xlabel('Kinetic Energy (eV)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()





# plot_atoms(atoms, radii=0.3, rotation=('20x,25y,20z'))





from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt
import numpy as np

# Load the trajectory
trajectory_filename = 'argon_trajectory.traj'
traj = Trajectory(trajectory_filename)

# List to store kinetic energies of each atom for all frames
kinetic_energies = []

# Loop over the frames in the trajectory
for atoms in traj:
    # Get velocities and masses
    velocities = atoms.get_velocities()
    masses = atoms.get_masses()
    # KE = 1/2 * m * v^2
    kinetic_energy_per_atom = 0.5 * masses[:, np.newaxis] * velocities ** 2
    kinetic_energy_per_atom = np.sum(kinetic_energy_per_atom, axis=1)
    
    
    # Add the kinetic energies of each atom in this frame to the main list
    kinetic_energies.append(kinetic_energy_per_atom)

kinetic_energies_flat = np.concatenate(kinetic_energies)








# Plot the histogram
plt.figure(figsize=(8, 6))
# plt.hist(kinetic_energies_flat, bins=200, color='green', alpha=0.7)

hist, bin_edges = np.histogram(kinetic_energies_flat, bins=200, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Define the Maxwell-Boltzmann distribution function
def maxwell_boltzmann(KE, T, scale):
    kbT = units.kB * T
    return scale * np.power(kbT, -3/2) * np.sqrt(KE) * np.exp(- KE / kbT) 
    

popt, _ = curve_fit(maxwell_boltzmann, bin_centers, hist, p0=[200 , np.amax(hist)])


plt.scatter(bin_centers, hist)
plt.plot(bin_centers, maxwell_boltzmann(bin_centers, *popt), 'r-', label=f'Fit (T = {popt[0]:.2f} K)')
plt.title('Histogram of Kinetic Energies: Maxwell-Boltzmann Distribution Fit')
plt.xlabel('Kinetic Energy per Atom (eV)')
plt.ylabel('Frequency')
plt.legend()
plt.show()








# %%

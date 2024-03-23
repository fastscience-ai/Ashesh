#!/usr/bin/env python
# coding: utf-8

# The MD simulation without any reactions, just the particles moving around with temperature
# 2024.03.15
# HongKee Yoon

# pip install numpy
# pip install ase 
# pip install matplotlib

# %%

import numpy as np
import ase
import ase.io
# Input
trajectory_filename = 'argon_trajectory.traj'

# %%
ase_traj = ase.io.read(trajectory_filename, index=':')
# %%
# print the first frame
# get atomic positions, cell, and atomic numbers
print("Postion\n",ase_traj[0].positions)
print("The Lattice\n",ase_traj[0].cell)
print("The atomic numbers\n",ase_traj[0].get_atomic_numbers())

# for every frame in the trajectory
# print the position, cell, and atomic numbers
# break after 3 frames
for i in range(0, len(ase_traj)):
    print("Frame", i)
    print("Postion\n",ase_traj[i].positions)
    print("The Lattice\n",ase_traj[i].cell)
    print("The atomic numbers\n",ase_traj[i].get_atomic_numbers())

    if i >= 3:
        break

# %%

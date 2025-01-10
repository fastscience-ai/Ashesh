#!/usr/bin/env python
# coding: utf-8

# The MD simulation without any reactions, just the particles moving around with temperature
# 2024.03.15
# HongKee Yoon

# pip install numpy
# pip install ase 
# pip install matplotlib

# %%
import os
import numpy as np
import ase
import ase.io
# Input
os.chdir('/scratch/x2895a03/research/md-diffusion/Ashesh')
temperature = '300K'
trajectory_filename = f'argon_trajectory_long_{temperature}.traj'


# %%
ase_traj = ase.io.read(trajectory_filename, index=':')
# %%
# print the first frame
# get atomic positions, cell, and atomic numbers
#print(ase_traj[0].get_forces)
print("Postion\n",ase_traj[0].positions, np.shape(ase_traj[0].positions), len(ase_traj))
#print("The Lattice\n",ase_traj[0].cell)
#print("The atomic numbers\n",ase_traj[0].get_atomic_numbers())
print("Postion\n",ase_traj[0].positions)
print("The Lattice\n",ase_traj[0].cell)
print("The atomic numbers\n",ase_traj[0].get_atomic_numbers())

# for every frame in the trajectory
# print the position, cell, and atomic numbers
# break after 3 frames
#(64, 3) 64
#[x,y,z, force_x,y,z, velocity_x,y,z, atom_type]
n,m = np.shape(ase_traj[0].positions)
output = []
print(len(ase_traj))
for i in range(0,  len(ase_traj)):
    inner_out = []
    for j in range(n):
        if i%100 == 0: print(i,j)
        pos = ase_traj[i].positions
        force = ase_traj[i].get_forces()
        vel =  ase_traj[i].get_velocities()
        inner_out.append([  pos[j][0], pos[j][1], pos[j][2], force[j][0], force[j][1], force[j][2], vel[j][0],  vel[j][1],  vel[j][2]])
    output.append(inner_out)
out = np.asarray(output)
d1,d2,d3 = np.shape(out)
print(d1,d2,d3) #500001 64 9
time_length = 10000
oo = np.shape(out[:-1])
print(oo)
input_ = np.reshape(out[:-1], (int(oo[0]/time_length),(time_length),oo[1],oo[2])) #ValueError: cannot reshape array of size 288000576 into shape (5000,100,64,9)
ii = np.shape(out[1:])
print(ii)
output_ = np.reshape(out[1:,:,:], (int(ii[0]/time_length),(time_length),ii[1],ii[2]))
print(np.shape(input_), np.shape(output_))
np.save(f"input_long_{temperature}.npy", input_)
np.save(f"output_long.npy_{temperature}", output_)
print(np.shape(output_), np.shape(input_))
   # print("Postion\n",ase_traj[i].positions)
   # print("Force\n", ase_traj[i].get_forces)
   # print("Velocity\n", ase_traj[i].get_velocities)
   #print("The Lattice\n",ase_traj[i].cell)
   # print("The atomic numbers\n",ase_traj[i].get_atomic_numbers())

for i in range(0, len(ase_traj)):
    print("Frame", i)
    print("Postion\n",ase_traj[i].positions)
    print("The Lattice\n",ase_traj[i].cell)
    print("The atomic numbers\n",ase_traj[i].get_atomic_numbers())

    if i >= 3:
        break

# %%

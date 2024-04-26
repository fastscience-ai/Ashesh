# %%
import numpy as np
import ase
import ase.io
import os
import tqdm
from ase.build import bulk
# %%
output_file = 'results/output.npy'
output_np = np.load(output_file) # (#timestep, #atoms, 9)
n_timestep = output_np.shape[0]
n_atoms = output_np.shape[1]
# %%
supercell_size = (4,4,4)
atoms = bulk('Ar', 'sc', a=5.26, cubic=True) * supercell_size # Create a 4x4x4 supercell

lattice = atoms.get_cell()  # 3x3 matrix
# %%
from ase.io import write

# extxyz 파일 경로 설정
output_file = 'results/trajectory.extxyz'

# 모든 timestep에 대해 반복
with open(output_file, 'w') as f:
    for i in tqdm.tqdm( range(n_timestep)):
        # 현재 timestep의 원자 위치 정보 추출
        positions = output_np[i, :, :3]
        
        # ASE Atoms 객체 생성
        atoms = ase.Atoms(symbols=['Ar'] * n_atoms, positions=positions, cell=lattice, pbc=True)
        
        # extxyz 형식으로 파일에 쓰기
        write(f, atoms, format='extxyz')

# %%
import numpy as np
from tqdm import tqdm

def calculate_msd(positions, timestep, lag_times):
    """
    Calculate the Mean Squared Displacement (MSD) from a trajectory for specified lag times.
    
    Args:
        positions (numpy.ndarray): Array of shape (n_timestep, n_atoms, 3) containing the positions of atoms.
        timestep (float): Time step between consecutive frames in the trajectory.
        lag_times (list): List of lag times to consider for MSD calculation.
    
    Returns:
        msd (numpy.ndarray): Array of shape (len(lag_times),) containing the MSD values for specified lag times.
    """
    n_timestep, n_atoms, _ = positions.shape
    msd = np.zeros(len(lag_times))
    
    for i, lag in enumerate(tqdm(lag_times)):
        sq_disp = []
        for j in range(n_timestep - lag):
            disp = positions[j + lag] - positions[j]
            sq_disp.append(np.sum(disp**2, axis=1))
        msd[i] = np.mean(sq_disp)
    
    msd *= timestep**2  # Convert squared displacement to squared distance
    
    return msd

# 예제 사용법
timestep = 1.0  # 시간 간격 (예: 1 femtosecond)
lag_times = [5,10, 20, 30, 50, 100, 200, 300]  # 사용자가 임의로 입력한 lag time 리스트

msd = calculate_msd(output_np[:, :, :3], timestep, lag_times)

# MSD 결과 출력 및 플롯
print("Mean Squared Displacement (MSD):")
for lag, msd_val in zip(lag_times, msd):
    print(f"Lag Time: {lag}, MSD: {msd_val}")

import matplotlib.pyplot as plt

plt.plot(lag_times, msd, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Lag Time')
plt.ylabel('Mean Squared Displacement')
plt.show()
# %%
# fit the MDS to the Einstein relation
# find the exponent of the MSD
from scipy.optimize import curve_fit
fit = np.polyfit(np.log(lag_times), np.log(msd), 1)
print(f"MSD exponent: {fit[0]}")
# plot the fit
plt.plot(lag_times, msd, marker='o', label='MSD')
plt.plot(lag_times, np.exp(fit[1]) * lag_times**fit[0], label='Fit MDS vs time '+str(fit[0]))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Lag Time')
plt.ylabel('Mean Squared Displacement')
plt.legend()
plt.savefig('results/msd.png')
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.random import default_rng
from point_neuron import PointNeuron
import os
import time
from helper import plot_loss,plot_field

# For 3D Point Neuron Learning stick to Automatic Gradient Descent using autograd functionality of Pytorch
use_manual = False

# Create Results Folder
mode = 'Manual' if use_manual else 'Auto'
results_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/Python/3-D/Results/trial'
data_dir    = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/Python/3-D/Data/Measurements/'
os.makedirs(results_dir, exist_ok=True)
# Set random seed for reproducibility
rng = default_rng(1)
# Load microphone coordinates and measurements of sound field
mic_data = loadmat(data_dir +'trial/MIC_DATA_test.mat')
MICCAR = mic_data['MicCoord']
MicPrimaryField = mic_data['MicData'].flatten()
MicN = MICCAR.shape[0]

# Load initial point neuron coordinates
pn_data = loadmat(data_dir + 'trial/PN_INITIALIZATION_test.mat')
InCoord = pn_data['biases']
PnN = InCoord.shape[0]

# Initialize PN weights
'''
If multiple sources this initialization works better.

InWeight = np.random.rand(PnN, MicN)
InWeight = InWeight / InWeight.sum(axis=1, keepdims=True)  # normalize rows
InWeight = InWeight @ MicPrimaryField  # Weighted avg mic response per PN
'''
InWeight = pn_data['weights'].squeeze()
#print(PnN,MicN)

# Parameters
Frequency = 900
c = 343
k = 2 * np.pi * Frequency / c
TargetR = 0.50
IterN = 20000
StepW = 0.005 #0.001
StepC = [0.005, 0.005, 0.005] #0.02
Lambda = 0.005 #0.01
L = (1.3,1.6,1.05) # room_dimensions
eps =1e-8


# Load ground truthdiff_planes
cir_data = loadmat(data_dir + 'trial/P_TARGET_test.mat')
pla_data = loadmat(data_dir + 'trial/P_ROOM_test.mat')
Pressure_room = pla_data['pressure_room'].flatten()
Samples_room = pla_data['sampling_room']
Samples_target = cir_data['sampling_target']
Pressure_target = cir_data['pressure_target'].flatten()
# Normalize
NormalFactor = np.linalg.norm(MicPrimaryField)
MicPrimaryField = MicPrimaryField/ NormalFactor
'''
Pressure_room = Pressure_room / NormalFactor
Pressure_target = Pressure_target / NormalFactor
'''
dx, dy, dz = 0.025,0.025,0
x = np.arange(0, L[0]+dx, dx)
y = np.arange(0, L[1]+dy, dy)
z = [0.525]
PxSampling, PySampling, PzSampling = np.meshgrid(x, y, z, indexing='ij')
PxN, PyN, PzN = len(x), len(y), len(z)
Estimated_Pressure = np.zeros(PxN * PyN * PzN, dtype=np.complex128)
Target_Pressure = []
Index = 0
# Point Neuron Learning (Manual or Auto)
model = PointNeuron(k, MicPrimaryField,MICCAR,InCoord, InWeight,StepW, StepC, Lambda, IterN)

start_time = time.time()
if use_manual:
    PnCoord, PnWeight, PnWscale, Loss = model.train_manual()
else:
    PnCoord, PnWeight, PnWscale, Loss = model.train_autograd()
end_time = time.time()

PnCoord = PnCoord.cpu().numpy()
PnWeight = PnWeight.cpu().numpy()
PnWscale = PnWscale.cpu().numpy()


# Save loss to CSV
loss_file = os.path.join(results_dir, 'loss.csv')
np.savetxt(loss_file, Loss, delimiter=',')
with open(os.path.join(results_dir, 'time.txt'), 'w') as f:
    f.write(f"{end_time - start_time:.2f}")

# Plot Loss Curve
plot_loss(Loss,results_dir)
plot_field(Samples_room, 'Source Sound Field', Pressure_room, f'{results_dir}/source_sound_field.png', clim=[-0.4, 0.4],z=0.525)

# Sound field estimation
for i in range(PxN):
    for j_idx in range(PyN):
        for k_idx in range(PzN):
            SampleV = np.array([PxSampling[i, j_idx, k_idx], PySampling[i, j_idx, k_idx], PzSampling[i, j_idx, k_idx]])
            WavePn = PnCoord.T  # (3, N_sources)
            DisPn = np.sum((WavePn - SampleV[:, None]) ** 2, axis=0)+eps
            sqrt_DisPn = np.sqrt(DisPn)
            G = np.exp(1j * k * sqrt_DisPn) / (4 * np.pi * sqrt_DisPn)
            Estimated_Pressure[Index] = NormalFactor*(PnWeight * PnWscale)@G
            Index += 1
EME = 10 * np.log10(np.abs(Pressure_room - Estimated_Pressure)**2 / (np.abs(Pressure_room)**2))
plot_field(Samples_room, 'Point Neuron Estimated Sound Field', Estimated_Pressure, f'{results_dir}/pn_estimated_field.png', clim=[-0.4, 0.4],z=0.525)
plot_field(Samples_room, 'Error Map', EME, f'{results_dir}/error_map.png', clim=[-15, 15], cmap='hot', cbar_ticks=[-15, -10, -5, 0, 5],z=0.525)
np.savetxt(f'{results_dir}/error_map.csv', EME, delimiter=',')








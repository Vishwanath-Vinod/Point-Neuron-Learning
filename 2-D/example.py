import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.random import default_rng
from point_neuron import PointNeuron
import os
import time
from helper import awgn, plot_field, plot_loss

# Toggle between manual and automatic computation of gradients
use_manual = False 

# Create Results Folder
mode = 'Manual' if use_manual else 'Auto'
results_dir = f'Results/{mode}'
os.makedirs('Results', exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Set random seed for reproducibility
rng = default_rng(1)

# Load microphone coordinates and measurements of sound field
mic_data = loadmat('Data/Mic75_random_Freq900.mat')
MICCAR = mic_data['MICCAR']
MicPrimaryField = mic_data['MicPrimaryField'].flatten()
MicN = MICCAR.shape[0]

# Load initial point neuron coordinates
pn_data = loadmat('Data/PNinital_Freq900.mat')
InCoord = pn_data['InCoord']
PnN = InCoord.shape[0]
#print(PnN,MicN)

# Parameters
Frequency = 900
c = 343
k = 2 * np.pi * Frequency / c
TargetR = 1
IterN = 10000
StepW = 0.03 #0.001
StepC = [0.005, 0.005, 0] #0.02
Lambda = 0.0005 #0.01
InWeight = rng.random(PnN)

# Load ground truth
cir_data = loadmat(f'Data/Cir1124_Freq{Frequency}.mat')
pla_data = loadmat(f'Data/Pla2500_Freq{Frequency}.mat')
PlaneSure = pla_data['PlaneSure'].flatten()
PLACAR = pla_data['PLACAR']
ValueSure = cir_data['ValueSure'].flatten()

# Add white Gaussian noise
MicPrimaryField_noisy = awgn(MicPrimaryField, 30)
NormalFactor = np.linalg.norm(MicPrimaryField_noisy)

# Point Neuron Learning (Manual or Auto)
model = PointNeuron(
    k, MicPrimaryField_noisy / NormalFactor,
    np.hstack((MICCAR, np.zeros((MicN, 1)))),
    np.hstack((InCoord, np.zeros((PnN, 1)))), InWeight,
    StepW, StepC, Lambda, IterN
)

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

# Sound field estimation
PxN, PyN = 50, 50
SizeX, SizeY = 1.3, 1.3
PxSampling = np.linspace(-SizeX, SizeX, PxN)
PySampling = np.linspace(-SizeY, SizeY, PyN)
PlanePn = np.zeros(PxN * PyN, dtype=np.complex128)
TargetPn = []
Index = 0
for i in range(PxN):
    for j in range(PyN):
        SampleV = np.array([PxSampling[i], PySampling[j]])
        WavePn = PnCoord[:, :2].T
        DisPn = np.sum((WavePn - SampleV[:, None]) ** 2, axis=0)
        PlanePn[Index] = (PnWeight * PnWscale) @ (
            np.exp(1j * k * np.sqrt(DisPn)) / (4 * np.pi * np.sqrt(DisPn))
        )
        if PxSampling[i]**2 + PySampling[j]**2 < TargetR**2:
            TargetPn.append(PlanePn[Index])
        Index += 1

TargetPn = np.array(TargetPn)
EME = 10 * np.log10(np.abs(PlaneSure - PlanePn)**2 / np.abs(PlaneSure)**2)
np.savetxt(f'{results_dir}/error_map.csv', EME, delimiter=',')

# Plot fields
plot_field(PLACAR, 'Source Sound Field', PlaneSure, f'{results_dir}/source_sound_field.png', clim=[-0.25, 0.25])
plot_field(PLACAR, 'Point Neuron Estimated Sound Field', PlanePn, f'{results_dir}/pn_estimated_field.png', clim=[-0.25, 0.25])
plot_field(PLACAR, 'Error Map', EME, f'{results_dir}/error_map.png', clim=[-15, 5], cmap='hot', cbar_ticks=[-15, -10, -5, 0, 5])

# Plot Loss Curve
plot_loss(Loss,results_dir)

# NSE calculation
ErrorPlane = 20 * np.log10(np.linalg.norm(PlaneSure - PlanePn) / np.linalg.norm(PlaneSure))
ErrorTarget = 20 * np.log10(np.linalg.norm(ValueSure - TargetPn) / np.linalg.norm(ValueSure))
print("NSE (entire plane):", ErrorPlane)
print("NSE (target region):", ErrorTarget)

# Point Neuron Learning

A from-scratch Python implementation of **Point Neuron Learning (PNL)** for narrowband sound-field reconstruction, extending the existing 2-D implementation by Hanwen to **3-D sound-field reconstruction**.

## Repository Structure

```text
Point-Neuron-Learning/
├── 2-D/
│   ├── Data/
│   │   ├── Cir1124_Freq900.mat
│   │   ├── Mic75_random_Freq900.mat
│   │   ├── PNinital_Freq900.mat
│   │   └── Pla2500_Freq900.mat
│   │
│   ├── Results/
│   │   ├── Autograd_Error_map.png
│   │   ├── loss_comparison.png
│   │   └── Manual_Backprop_error_map.png
│   │
│   ├── example.py
│   ├── helper.py
│   └── point_neuron.py
│
├── 3-D/
│   ├── Code/
│   │   ├── Twodexample.py
│   │   ├── example.py
│   │   ├── helper.py
│   │   ├── point_neuron.py
│   │   └── test.py
│   │
│   └── Data/
│       └── Matlab_Code/
│           ├── RIR.m
│           ├── ism.m
│           ├── sample_circular_region.m
│           ├── sample_spherical_region.m
│           ├── structured_bias_mesh.m
│           └── uniform_spherical_region.m
│
├── requirements.txt
└── README.md
```

## Overview

**Point Neuron Learning** represents a sound field using a set of learnable point sources. The spatial locations and complex-valued weights of the point neurons are optimized to reconstruct sound-field measurements obtained from a microphone array.

This repository contains:

* A **2-D implementation** of Point Neuron Learning.
* A **3-D extension** for three-dimensional sound-field reconstruction.
* Implementations using **PyTorch automatic differentiation** and manual backpropagation.
* MATLAB utilities for generating and sampling 3-D acoustic environments.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Vishwanath-Vinod/Point-Neuron-Learning.git
cd Point-Neuron-Learning
pip install -r requirements.txt
```

## Usage

### 2-D

The 2-D implementation is located in `2-D/`.

```bash
cd 2-D
python example.py
```

The implementation and supporting utilities are provided in:

```text
point_neuron.py
helper.py
example.py
```

### 3-D

The 3-D implementation is located in `3-D/Code/`.

```bash
cd 3-D/Code
python example.py
```

Additional scripts are provided for testing and 2-D comparisons.

## Results

The `2-D/Results/` directory contains comparisons between **PyTorch autograd** and **manual backpropagation**, including reconstruction error maps and loss curves.

## Data

The repository includes example microphone-array measurements, microphone positions, point-neuron initializations, and related acoustic data in the `Data/` directories.

The `3-D/Data/Matlab_Code/` directory contains MATLAB scripts for generating room impulse responses and sampling circular and spherical regions used in the 3-D experiments.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{vinodpoint,
  title={Point Neuron Embedded Kalman Filter for Narrowband Sound Source Tracking},
  author={Vinod, Vishwanath and Xu, Shaoheng and Samarasinghe, Prasanga N and Bastine, Amy and Abhayapala, Thushara D}
}
```

**Vishwanath Vinod**
Indian Institute of Technology Madras

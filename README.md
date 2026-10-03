# Point Neuron Learning

A from-scratch Python implementation of **Point Neuron Learning (PNL)** for sound-field reconstruction, including an extension to 3D from the existing 2D implementation by Hanwen.

## Repository Structure

```text
Point-Neuron-Learning/
├── 2-D/
│   ├── Data/ (Microphone data and positions, Point Neuron Locations and Weights)
│   │   ├── Cir1124_Freq900.mat
│   │   ├── Mic75_random_Freq900.mat
│   │   ├── PNinital_Freq900.mat
│   │   └── Pla2500_Freq900.mat
│   ├── Results/ (Comparing PyTorch Autograd & Manual Backpropagation from the Point Neuron Learning Paper)
│   │   ├── Autograd_Error_map.png
│   │   ├── loss_comparison.png
│   │   └── Manual_Backprop_error_map.png
│   ├── example.py
│   ├── helper.py
│   ├── point_neuron.py #Python Implementation of 2-D Point Neuron Learning
├── 3-D
│   ├── Code # Python Implementation of 3-D Point Neuron Learning
│   │   ├── Twodexample.py
│   │   ├── example.py
│   │   ├── helper.py
│   │   ├── point_neuron.py
│   │   └── test.py
│   ├── Data
│   │   ├── Matlab_Code
│   │   │   ├── RIR.m
│   │   │   ├── ism.m
│   │   │   ├── sample_circular_region.m
│   │   │   ├── sample_spherical_region.m
│   │   │   ├── structured_bias_mesh.m
│   │   │   └── uniform_spherical_region.m
└── requirements.txt
└── README.md
```

## Overview

Point Neuron Learning represents an acoustic field using a set of learnable point sources. The locations and complex-valued weights of the point neurons are optimized to reconstruct sound field measurements from a microphone array.

This repository contains:

* A **2-D implementation** of Point Neuron Learning
* A **3-D extension** for reconstructing 3-dimensional sound fields
* Python implementations of the optimisation and reconstruction procedure

## Installation

```bash
git clone https://github.com/Vishwanath-Vinod/Point-Neuron-Learning.git
cd Point-Neuron-Learning
pip install -r requirements.txt
```

Run the corresponding scripts from each directory according to the experimental configuration.

## Citation
@article{vinodpoint,
  title={Point Neuron Embedded Kalman Filter for Narrowband Sound Source Tracking},
  author={Vinod, Vishwanath and Xu, Shaoheng and Samarasinghe, Prasanga N and Bastine, Amy and Abhayapala, Thushara D}
}

# Point Neuron Learning

A from-scratch Python implementation of **Point Neuron Learning (PNL)** for sound-field reconstruction, including an extension to 3D from the existing 2D implementation by Hanwen.

## Repository Structure

```text
Point-Neuron-Learning/
.
├── 2-D
│   ├── Data (Microphone data and Point Neuron Locations and Weights
│   │   ├── Cir1124_Freq900.mat
│   │   ├── Mic75_random_Freq900.mat
│   │   ├── PNinital_Freq900.mat
│   │   └── Pla2500_Freq900.mat
│   ├── Results-2D (Comparing Pytorch Autograd & Manual Backpropagation from the Point Neuron Learning Paper)
│   │   ├── Auto
│   │   │   ├── error_map.csv
│   │   │   ├── error_map.png
│   │   │   ├── loss.csv
│   │   │   ├── pn_estimated_field.png
│   │   │   ├── source_sound_field.png
│   │   │   ├── time.txt
│   │   │   └── zoomed_loss.png
│   │   ├── Compare
│   │   │   └── loss_comparison.png
│   │   └── Manual
│   │       ├── error_map.csv
│   │       ├── error_map.png
│   │       ├── loss.csv
│   │       ├── pn_estimated_field.png
│   │       ├── source_sound_field.png
│   │       ├── time.txt
│   │       └── zoomed_loss.png
│   ├── example.py
│   ├── helper.py
│   ├── point_neuron.py
├── 3-D
│   ├── Code
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
│   │   └── Measurements
│   │       ├── 2d
│   │       │   ├── beta=0
│   │       │   │   ├── MIC_DATA_2D.mat
│   │       │   │   ├── PN_INITIALIZATION_2D.mat
│   │       │   │   ├── P_ROOM_2D.mat
│   │       │   │   └── P_TARGET_2D.mat
│   │       │   ├── beta=0.1
│   │       │   │   ├── MIC_DATA_2D.mat
│   │       │   │   ├── PN_INITIALIZATION_2D.mat
│   │       │   │   ├── P_ROOM_2D.mat
│   │       │   │   └── P_TARGET_2D.mat
│   │       │   └── beta=0.8
│   │       │       ├── MIC_DATA_2D.mat
│   │       │       ├── PN_INITIALIZATION_2D.mat
│   │       │       ├── P_ROOM_2D.mat
│   │       │       └── P_TARGET_2D.mat
│   │       ├── 3d
│   │       │   ├── planar_mics
│   │       │   │   ├── MIC_DATA_2D.mat
│   │       │   │   ├── PN_INITIALIZATION_2D.mat
│   │       │   │   ├── P_ROOM_2D.mat
│   │       │   │   ├── P_TARGET_2D.mat
│   │       │   │   └── orientation.png
│   │       │   └── spherical_mics
│   │       │       ├── num_sources=1
│   │       │       │   ├── 1source.png
│   │       │       │   ├── z=0.1(outside)
│   │       │       │   │   ├── MIC_DATA_2D.mat
│   │       │       │   │   ├── PN_INITIALIZATION_2D.mat
│   │       │       │   │   ├── P_ROOM_2D.mat
│   │       │       │   │   └── P_TARGET_2D.mat
│   │       │       │   ├── z=0.525
│   │       │       │   │   ├── MIC_DATA_2D.mat
│   │       │       │   │   ├── PN_INITIALIZATION_2D.mat
│   │       │       │   │   ├── P_ROOM_2D.mat
│   │       │       │   │   └── P_TARGET_2D.mat
│   │       │       │   └── z=0.625(off_center)
│   │       │       │       ├── MIC_DATA_2D.mat
│   │       │       │       ├── PN_INITIALIZATION_2D.mat
│   │       │       │       ├── P_ROOM_2D.mat
│   │       │       │       └── P_TARGET_2D.mat
│   │       │       ├── num_sources=2
│   │       │       │   ├── diff_planes
│   │       │       │   │   ├── MIC_DATA_2D.mat
│   │       │       │   │   ├── PN_INITIALIZATION_2D.mat
│   │       │       │   │   ├── P_ROOM_2D.mat
│   │       │       │   │   └── P_TARGET_2D.mat
│   │       │       │   ├── random_locations
│   │       │       │   │   ├── MIC_DATA_2D.mat
│   │       │       │   │   ├── PN_INITIALIZATION_2D.mat
│   │       │       │   │   ├── P_ROOM_2D.mat
│   │       │       │   │   ├── P_TARGET_2D.mat
│   │       │       │   │   └── mic_source_pn_plot.png
│   │       │       │   └── same_plane
│   │       │       │       ├── MIC_DATA_2D.mat
│   │       │       │       ├── PN_INITIALIZATION_2D.mat
│   │       │       │       ├── P_ROOM_2D.mat
│   │       │       │       └── P_TARGET_2D.mat
│   │       │       ├── num_sources=3
│   │       │       │   ├── diff_planes
│   │       │       │   │   ├── MIC_DATA_2D.mat
│   │       │       │   │   ├── PN_INITIALIZATION_2D.mat
│   │       │       │   │   ├── P_ROOM_2D.mat
│   │       │       │   │   ├── P_TARGET_2D.mat
│   │       │       │   │   └── mic_source_pn_plot.png
│   │       │       │   └── random_locations
│   │       │       │       ├── MIC_DATA_2D.mat
│   │       │       │       ├── PN_INITIALIZATION_2D.mat
│   │       │       │       ├── P_ROOM_2D.mat
│   │       │       │       ├── P_TARGET_2D.mat
│   │       │       │       └── mic_source_pn_plot.png
│   │       │       └── num_sources=5
│   │       │           ├── diff_planes
│   │       │           │   ├── MIC_DATA_2D.mat
│   │       │           │   ├── PN_INITIALIZATION_2D.mat
│   │       │           │   ├── P_ROOM_2D.mat
│   │       │           │   └── P_TARGET_2D.mat
│   │       │           └── random_locations
│   │       │               ├── MIC_DATA_2D.mat
│   │       │               ├── PN_INITIALIZATION_2D.mat
│   │       │               ├── P_ROOM_2D.mat
│   │       │               ├── P_TARGET_2D.mat
│   │       │               └── mic_source_pn_plot.png
│   │       └── trial
│   │           ├── MIC_DATA_2D.mat
│   │           ├── MIC_DATA_test.mat
│   │           ├── PN_INITIALIZATION_2D.mat
│   │           ├── PN_INITIALIZATION_test (1).mat
│   │           ├── PN_INITIALIZATION_test (2).mat
│   │           ├── PN_INITIALIZATION_test.mat
│   │           ├── P_ROOM_2D.mat
│   │           ├── P_ROOM_test.mat
│   │           ├── P_TARGET_2D.mat
│   │           ├── P_TARGET_test.mat
│   │           ├── mic_source_pn_plot_2D.png
│   │           └── mic_source_pn_plot_test.png
│   └── Results
│       ├── 2d
│       │   ├── Hanwen_data
│       │   │   └── Auto
│       │   │       ├── error_map.csv
│       │   │       ├── error_map.png
│       │   │       ├── loss.csv
│       │   │       ├── pn_estimated_field.png
│       │   │       ├── source_sound_field.png
│       │   │       ├── time.txt
│       │   │       └── zoomed_loss.png
│       │   ├── beta=0
│       │   │   ├── error_map.csv
│       │   │   ├── error_map.png
│       │   │   ├── loss.csv
│       │   │   ├── loss.png
│       │   │   ├── magnitude_comparison.png
│       │   │   ├── pn_estimated_field.png
│       │   │   ├── source_sound_field.png
│       │   │   └── time.txt
│       │   ├── beta=0.1
│       │   │   ├── error_map.csv
│       │   │   ├── error_map.png
│       │   │   ├── loss.csv
│       │   │   ├── loss.png
│       │   │   ├── pn_estimated_field.png
│       │   │   ├── source_sound_field.png
│       │   │   └── time.txt
│       │   └── beta=0.8
│       │       ├── error_map.csv
│       │       ├── error_map.png
│       │       ├── loss.csv
│       │       ├── loss.png
│       │       ├── pn_estimated_field.png
│       │       ├── source_sound_field.png
│       │       └── time.txt
│       ├── 3d
│       │   ├── planar_mics
│       │   │   ├── error_map.csv
│       │   │   ├── error_map.png
│       │   │   ├── loss.csv
│       │   │   ├── loss.png
│       │   │   ├── pn_estimated_field.png
│       │   │   ├── source_sound_field.png
│       │   │   └── time.txt
│       │   └── spherical_mics
│       │       ├── num_sources=1
│       │       │   └── pns_same_plane_as_source
│       │       │       ├── z=0.1(outside)
│       │       │       │   ├── error_map.csv
│       │       │       │   ├── error_map.png
│       │       │       │   ├── loss.csv
│       │       │       │   ├── loss.png
│       │       │       │   ├── pn_estimated_field.png
│       │       │       │   ├── source_sound_field.png
│       │       │       │   └── time.txt
│       │       │       ├── z=0.525
│       │       │       │   ├── error_map.csv
│       │       │       │   ├── error_map.png
│       │       │       │   ├── loss.csv
│       │       │       │   ├── loss.png
│       │       │       │   ├── pn_estimated_field.png
│       │       │       │   ├── source_sound_field.png
│       │       │       │   └── time.txt
│       │       │       └── z=0.625(off_center)
│       │       │           ├── error_map.csv
│       │       │           ├── error_map.png
│       │       │           ├── loss.csv
│       │       │           ├── loss.png
│       │       │           ├── pn_estimated_field.png
│       │       │           ├── source_sound_field.png
│       │       │           └── time.txt
│       │       ├── num_sources=2
│       │       │   ├── pns_randomly_placed
│       │       │   │   ├── error_map.csv
│       │       │   │   ├── error_map.png
│       │       │   │   ├── loss.csv
│       │       │   │   ├── loss.png
│       │       │   │   ├── pn_estimated_field.png
│       │       │   │   ├── source_sound_field.png
│       │       │   │   └── time.txt
│       │       │   ├── pns_same_plane_as_source
│       │       │   │   ├── error_map.csv
│       │       │   │   ├── error_map.png
│       │       │   │   ├── loss.csv
│       │       │   │   ├── loss.png
│       │       │   │   ├── pn_estimated_field.png
│       │       │   │   ├── source_sound_field.png
│       │       │   │   └── time.txt
│       │       │   └── sources_on_same_plane
│       │       │       ├── error_map.csv
│       │       │       ├── error_map.png
│       │       │       ├── loss.csv
│       │       │       ├── loss.png
│       │       │       ├── pn_estimated_field.png
│       │       │       ├── source_sound_field.png
│       │       │       └── time.txt
│       │       ├── num_sources=3
│       │       │   ├── pns_randomly_placed
│       │       │   │   ├── error_map.csv
│       │       │   │   ├── error_map.png
│       │       │   │   ├── loss.csv
│       │       │   │   ├── loss.png
│       │       │   │   ├── pn_estimated_field.png
│       │       │   │   ├── source_sound_field.png
│       │       │   │   └── time.txt
│       │       │   └── pns_same_plane_as_source
│       │       │       ├── error_map.csv
│       │       │       ├── error_map.png
│       │       │       ├── loss.csv
│       │       │       ├── loss.png
│       │       │       ├── pn_estimated_field.png
│       │       │       ├── source_sound_field.png
│       │       │       └── time.txt
│       │       └── num_sources=5
│       │           ├── pns_randomly_placed
│       │           │   ├── error_map.csv
│       │           │   ├── error_map.png
│       │           │   ├── loss.csv
│       │           │   ├── loss.png
│       │           │   ├── pn_estimated_field.png
│       │           │   ├── source_sound_field.png
│       │           │   └── time.txt
│       │           └── pns_same_plane_as_source
│       │               ├── error_map.csv
│       │               ├── error_map.png
│       │               ├── loss.csv
│       │               ├── loss.png
│       │               ├── pn_estimated_field.png
│       │               ├── source_sound_field.png
│       │               └── time.txt
│       ├── freeze_biases
│       │   ├── error_map.csv
│       │   ├── error_map.png
│       │   ├── loss.csv
│       │   ├── loss.png
│       │   ├── pn_estimated_field.png
│       │   ├── source_sound_field.png
│       │   └── time.txt
│       └── trial
│           ├── error_map.csv
│           ├── error_map.png
│           ├── loss.csv
│           ├── loss.png
│           ├── pn_estimated_field.png
│           ├── source_sound_field.png
│           └── time.txt
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

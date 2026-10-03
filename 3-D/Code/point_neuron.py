# Creation   : 28-05-2025
# Author: Vishwanath Vinod
# Description:
#       The function iteratively updates the locations
#       and weights of point neurons by gradient descent.
#
# Shape:
#       1) P: Number of Point Neurons (Virtual sources) determined by the InCoord and InWeight shape
#       2) Q: Number of Microphones given by the shape of MicField and MicCoord
#
# Inputs:
#       1) k        : The wavenumber.
#       2) MicField : The microphone measured sound field in the frequency 
#                     domain (Q-by-1).
#       3) MicCoord : the Cartesian coordinates of the microphone points 
#                     (Q-by-3).
#       4) InCoord  : the Cartesian coordinates of the initial neurons 
#                     (P-by-3).
#       5) InWeight : the initial weight of point neurons (P-by-1).
#       6) StepC    : Step size for updating point neuron location 
#                     in xyz (3-by-1).
#       7) StepW    : Step size for updating point neuron weights.
#       8) Lambda   : Model complexity penalty (L1 norm).
#       9) IterN    : No of iterations.
#
# 
# Outputs:
#       1) PnCoord  : The optimal Cartesian location of point neurons 
#                     (P-by-3).
#       2) PnWeight : The optimal weights of point neurons (P-by-3).
#       3) PnWscale : Scale parts of the mix-wave function (P-by-1).
#       4) Loss     : Loss of the network.


#########################################################################

import torch
import numpy as np
from tqdm import tqdm

class PointNeuron:
    def __init__(self, k, MicField, MicCoord, InCoord, InWeight, StepW, StepC, Lambda, IterN):
        '''
        Initializing the input parameters and other network settings similar to original code
        '''
        self.k = k
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MicField = torch.tensor(MicField, dtype=torch.cfloat,device=self.device)   #(Q,)
        self.MicCoord = torch.tensor(MicCoord, dtype=torch.float32,device=self.device)  #(Q,3)
        self.InCoord = torch.tensor(InCoord, dtype=torch.float32,device=self.device)    #(P,3)
        self.InWeight = torch.tensor(InWeight, dtype=torch.cfloat,device=self.device)   #(P,)
        
        self.StepW = StepW
        self.StepC = StepC        # (3,) step sizes for x, y, z
        self.Lambda = Lambda
        self.IterN = IterN
        self.P = InCoord.shape[0]
        self.Q = MicCoord.shape[0]


        self.grad_b = 0.3
        self.loss_threshold = 5e-7
        self.optimal_loss = 2e7

    def calculate_loss(self,pn_coord,pn_weight,mic_pred,scale):
        '''
        Calculate the loss function given a set of locations and strengths for each microphone
        '''
        loss = torch.sum(torch.abs(mic_pred - self.MicField) ** 2) +self.Lambda * torch.sum(torch.abs(pn_weight))
        if loss.detach().item() < self.optimal_loss:
            self.optimal_loss = loss.detach().item()
            self.best_coord = pn_coord.detach().clone()
            self.best_weight = pn_weight.detach().clone()
            self.best_scale = scale[:, 0].detach().clone()
        
        return loss
    
    def train_autograd(self):
        '''
        Updating bias and weights of each neuron via backpropagation.
        Gradients are calculated using torch.autograd.
        '''
        # Initialize base parameters without grad
        bias = torch.tensor(self.InCoord, dtype=torch.float32, device=self.device).clone()
        weight = torch.tensor(self.InWeight, dtype=torch.cfloat, device=self.device).clone()
        loss_list = []
        self.best_scale = torch.zeros(self.P, dtype=torch.cfloat, device=self.device)

        for i in tqdm(range(self.IterN)):
            # Create new tensors with grad enabled from current parameters
            pn_coord = bias.clone().detach().requires_grad_(True)
            pn_weight = weight.clone().detach().requires_grad_(True)
            diff = pn_coord[:, None, :] - self.MicCoord[None, :, :]  # (P, Q, 3)
            dist_sq = torch.sum(diff ** 2, dim=2)
            dist = torch.sqrt(dist_sq)  # (P, Q)
            #print("distance",torch.min(dist))
            hn = torch.exp(1j * self.k * dist) / ((dist) * 4 * np.pi)  # (P, Q)
            dist_source = torch.norm(pn_coord, dim=1, keepdim=True).repeat(1, self.Q)
            #print("distance source",torch.min(dist_source))
            scale = dist_source * torch.exp(-1j * self.k * dist_source)  # (P, Q)
            mic_pred = torch.sum(pn_weight[:, None] * hn * scale, dim=0)  # (Q,)
            loss = self.calculate_loss(pn_coord, pn_weight, mic_pred, scale)
            loss_list.append(loss.item())
            
            if i > 0 and i % 100 == 0:
                print(f"Loss in {i}th iteration is: {loss.item()}")

            # Backpropagation
            loss.backward()
            #torch.nn.utils.clip_grad_norm_([pn_coord, pn_weight], max_norm=10.0)
            with torch.no_grad():
                max_delta = 0.2 * self.grad_b
                for p in range(self.P):
                    # Update weights (complex)
                    grad_w = pn_weight.grad[p]
                    if grad_w is not None:
                        weight[p] -= self.StepW * grad_w

                    # Update coords (real)
                    for j in range(3):
                        grad_c = pn_coord.grad[p, j]
                        if grad_c is not None:
                            delta = self.StepC[j] * grad_c.item()
                            if abs(delta) > max_delta:
                                delta = max_delta * delta / abs(delta)
                            bias[p, j] -= delta  
        return self.best_coord, self.best_weight, self.best_scale, loss_list
    
    def train_freeze_biases(self):
        '''
        Updating only weights of each neuron via backpropagation.
        Bias (coordinates) are frozen.
        Gradients are calculated using torch.autograd.
        '''
        # Freeze bias
        bias = torch.tensor(self.InCoord, dtype=torch.float32, device=self.device).clone()  # No requires_grad
        weight = torch.tensor(self.InWeight, dtype=torch.cfloat, device=self.device).clone()
        loss_list = []
        self.best_scale = torch.zeros(self.P, dtype=torch.cfloat, device=self.device)

        for i in tqdm(range(self.IterN)):
            # Only weights require gradients
            pn_coord = bias.clone()  # frozen, no grad
            pn_weight = weight.clone().detach().requires_grad_(True)

            # Compute distances and Green's function
            diff = pn_coord[:, None, :] - self.MicCoord[None, :, :]  # (P, Q, 3)
            dist = torch.norm(diff, dim=2)  # (P, Q)
            hn = torch.exp(1j * self.k * dist) / (dist * 4 * np.pi + 1e-8)

            dist_source = torch.norm(pn_coord, dim=1, keepdim=True).repeat(1, self.Q)
            scale = dist_source * torch.exp(-1j * self.k * dist_source)  # (P, Q)

            mic_pred = torch.sum(pn_weight[:, None] * hn * scale, dim=0)  # (Q,)
            loss = self.calculate_loss(pn_coord, pn_weight, mic_pred, scale)
            loss_list.append(loss.item())

            if i > 0 and i % 100 == 0:
                print(f"Loss at iteration {i}: {loss.item()}")

            # Backpropagation
            loss.backward()
            with torch.no_grad():
                for p in range(self.P):
                    grad_w = pn_weight.grad[p]
                    if grad_w is not None:
                        weight[p] -= self.StepW * grad_w  # Update weights only

        return self.best_coord, self.best_weight, self.best_scale, loss_list


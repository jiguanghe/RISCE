# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:43:31 2021

@author: jhe
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torch.autograd import Variable

N_r, N_t = 16, 1 
N_RIS = 32

M_t, M_r, M_RIS = 1, 8, 28

N_train = 100000
N_test = 10000
BATCH_SIZE = 64
L_RM = 1
L_BR = 1


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W


W1  = torch.from_numpy(DFT_matrix(N_r)[0:M_r,:]/np.sqrt(N_r))
Omega1  = torch.from_numpy(DFT_matrix(N_RIS)[:,0:M_RIS] )



def data_generate(N_t, N_r, N_RIS, M_t, M_r, M_RIS, N, L_BR, L_RM, snr):
    W = torch.zeros(N, M_r, N_r, dtype=torch.cfloat)
    Omega = torch.zeros(N, N_RIS, M_RIS, dtype=torch.cfloat)
    H_RM = torch.zeros(N,N_RIS, dtype=torch.cfloat)
    H_BR = torch.zeros(N,N_r, N_RIS, dtype=torch.cfloat)
    H = torch.zeros(N,N_r, N_RIS, dtype=torch.cfloat)
    h = torch.zeros(N, 1, 2*N_r*N_RIS)
    Y = torch.zeros(N,M_r, M_RIS, dtype=torch.cfloat)
    y = torch.zeros(N, 1, 2* M_r*M_RIS)
    A = torch.zeros(N, 2*M_r*M_RIS, 2*N_RIS*N_r)
    for i in range(N):
        W[i,:,:] = W1
        Omega[i,:,:] = Omega1
        f_RM = 1.0*torch.rand(L_RM)
        A_RM =  torch.vander(torch.exp(1j*math.pi*f_RM), N_RIS, increasing=True)
        rho_RM = torch.randn(L_RM, dtype=torch.cfloat)
        H_RM[i,:] = torch.matmul(torch.transpose(A_RM, 0, 1), rho_RM) 
        f1_BR = 1.0*torch.rand(L_BR)
        f2_BR = 1.0*torch.rand(L_BR)
        A1_BR = torch.vander(torch.exp(1j*math.pi*f1_BR), N_r, increasing=True)
        A2_BR = torch.vander(torch.exp(1j*math.pi*f2_BR), N_RIS, increasing=True)
        rho_BR = torch.randn(L_BR, dtype=torch.cfloat)
        H_BR[i,:,:]  =  torch.matmul( torch.matmul(torch.transpose(A1_BR, 0, 1), torch.diag(rho_BR)), A2_BR)
        H[i,:,:] = torch.matmul(H_BR[i,:,:], torch.diag(H_RM[i,:]))
        Y[i,:,:] =  torch.matmul(W[i,:,:], torch.matmul(H[i,:,:], Omega[i,:,:])) + 1/np.sqrt(10**(snr[i]/10))*torch.matmul(W[i,:,:], torch.randn(N_r, M_RIS, dtype=torch.cfloat))
        h1 = torch.reshape(torch.transpose(H[i,:,:],0,1),(1,-1))
        h[i,:,:] = torch.cat((h1.real, h1.imag),1)#/torch.norm(torch.cat((h1.real, h1.imag),1))
        A1 = torch.from_numpy(np.kron(torch.transpose(Omega[i,:,:], 0, 1), W[i,:,:]))
        A[i,:,:] =  torch.cat((torch.cat((A1.real, - A1.imag),1), torch.cat((A1.imag,A1.real),1)),0)
        y1 = torch.reshape(torch.transpose(Y[i,:,:],0,1),(1,-1))
        y[i,:,:] = torch.cat((y1.real, y1.imag),1)
    return  y,A,h


class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(2*N_RIS*N_r, 2*N_RIS*N_r) for i in range(15)])
        self.predict1 = nn.Linear(2*N_RIS*N_r,2*N_RIS*N_r)
        self.predict2 = nn.Linear(32,8)
        self.predict3 = nn.Linear(8,32)
        self.predict4 = nn.Linear(32,2*N_RIS*N_r)
        mus = torch.ones(15,3)*0.001
        sigma1 = torch.ones(11,1)
        sigma2 = torch.ones(1)
        self.Mus = torch.nn.Parameter(mus,requires_grad=True)
        self.sigma1 = torch.nn.Parameter(sigma1,requires_grad=True)
        self.sigma2 = torch.nn.Parameter(sigma2,requires_grad=True)
   
   
    def forward(self, h, A, y):
         for i in range(len(self.linears)):
             self.Mus.data = torch.clamp(self.Mus.data, min=0, max=1)
             self.sigma1.data = torch.clamp(self.sigma1.data, min=1, max=10)
             h = F.relu(self.linears[i](h) + self.Mus[i,0] * torch.transpose(torch.matmul(torch.transpose(A,1,2), torch.transpose(y,1,2)),1,2)- self.Mus[i,1] *torch.transpose(torch.matmul( torch.matmul(torch.transpose(A,1,2),A ),torch.transpose(h,1,2)),1,2) - self.Mus[i,2] *h) #/torch.norm(h+0.001)
         
         h = (self.predict1(h))
         return h

my_nn = simpleNet()

my_nn.to(device)

snr = 20.0*torch.ones([N_train,1]) 
y_train, A_train, h_train = data_generate(N_t, N_r, N_RIS, M_t, M_r, M_RIS, N_train, L_BR, L_RM,snr)

 
#*torch.rand([1])
torch_dataset = Data.TensorDataset(y_train, A_train, h_train)
loader = Data.DataLoader(
    dataset= torch_dataset,      # torch TensorDataset format
    batch_size= BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=0,              # subprocesses for loading data
)


def loss_func(output, target):
    loss = torch.norm(output - target)**2/torch.norm(target)**2
    return loss.mean()

optimizer = optim.Adam(my_nn.parameters(), lr = 0.0005, betas=(0.92, 0.99), eps=1e-08)
  
for epoch in range(40):
    if epoch > 20:
        lr = 0.0005

        for step, (batch_y, batch_A, batch_h) in enumerate(loader): 
            b_y = Variable(batch_y)
            b_A = Variable(batch_A)
            b_h = Variable(batch_h)
            b_y, b_A, b_h = b_y.to(device), b_A.to(device), b_h.to(device) 
            my_nn.zero_grad()
            h = torch.zeros([1, 1, 2*N_RIS*N_r])
            h = h.to(device)
            output = my_nn(h, b_A, b_y)
            loss = loss_func(output, b_h) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss)
# 

loss_pre = []  


my_nn.eval()
with torch.no_grad():    
    for k in np.linspace(0,20,5):
        y_test,A_test, h_test = data_generate(N_t, N_r, N_RIS, M_t, M_r, M_RIS, N_test, L_BR, L_RM,k*torch.ones([N_test,1]))
        torch_dataset = Data.TensorDataset(y_test, A_test, h_test)
        loader = Data.DataLoader(
            dataset= torch_dataset,      # torch TensorDataset format
            batch_size= BATCH_SIZE,      # mini batch size
            shuffle=True,               # random shuffle for training
            num_workers=0,              # subprocesses for loading data
        )


        for step, (batch_y, batch_A, batch_h) in enumerate(loader): 
            b_y = Variable(batch_y)
            b_A = Variable(batch_A)
            b_h = Variable(batch_h)
            b_y, b_A, b_h = b_y.to(device), b_A.to(device), b_h.to(device) 
            h = torch.zeros([1, 1, 2*N_RIS*N_r])
            h = h.to(device)
            output = my_nn(h, b_A, b_y)         
            loss = loss_func(output, b_h) 

        loss_pre.append(loss)
print(loss_pre)
loss_pre1 = torch.FloatTensor(loss_pre)
np.savetxt('np_loss_pre.txt',loss_pre1.cpu())

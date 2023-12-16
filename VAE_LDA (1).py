#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import trange
import scipy.io as sc

assert pyro.__version__.startswith('1.8.4')
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ


# In[2]:


# Read in the single cell count matrix which is a sparse matrix.
matrix = sc.mmread("all_cells.mtx")
sc_count = matrix.toarray() # Convert to dense matrix
sc_count = torch.tensor(sc_count).T
library = torch.sum(sc_count, dim = 1) # total counts of genes in each cell.


# In[3]:


class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, gene_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(gene_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs. 
        logphi_loc = self.bnmu(self.fcmu(h))   # phi denotes the topic prevalence within the cell.
        logphi_logvar = self.bnlv(self.fclv(h))
        logphi_scale = (0.5 * logphi_logvar).exp()  # Enforces positivity
        return logphi_loc, logphi_scale
    
        
class VAE_LDA(nn.Module):
    def __init__(self, gene_size, num_topics, hidden, dropout):
        super().__init__()
        self.gene_size = gene_size
        self.num_topics = num_topics
        self.encoder = Encoder(gene_size, num_topics, hidden, dropout)
        
    def model(self, sc_count):   # The generative model
        eta = torch.ones(num_topics)
        # With the following step, it generate a matrix where per gene is assigned to only one topics.
        with pyro.plate("gene", self.gene_size):
            orig_psi = pyro.sample("orig_psi", dist.Multinomial(probs = eta))  # psi denotes the distribution of genes in the topics.
        
        # Following steps are performed to ensure that the non-zero probabilities of genes in each topic are extracted from LogNormal distribution.
        mark = (orig_psi != 0)
        psi = torch.zeros_like(orig_psi)
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
        for col_idx in pyro.plate("topic", num_topics):
            col_mark = mark[:, col_idx]
            num_nonzeros = col_mark.sum().item()
            
            if num_nonzeros>0:
                probs = pyro.sample("probs_{}".format(col_idx), dist.LogNormal(mean, std).expand_by((num_nonzeros,)).to_event(1) )
                norm_probs = F.softmax(probs, dim = -1)
                psi[col_mark, col_idx] = norm_probs
       
        psi = psi.T  # Generate latent variable psi
        
        max_library_size = int(torch.max(library).item())
        with pyro.plate("cell", sc_count.shape[0]):
            logphi_loc = sc_count.new_zeros((sc_count.shape[0], num_topics))
            logphi_scale = sc_count.new_ones((sc_count.shape[0], num_topics))
            logphi = pyro.sample(
                "logphi", dist.Normal(logphi_loc, logphi_scale).to_event(1))
            phi = F.softmax(logphi, -1)   # Generate latent variable phi
            pyro.sample("obs", dist.Multinomial(max_library_size, phi@psi), obs=sc_count) # Likelihood
    
    
    def guide(self, sc_count):    # Approximation of the posterior distribution
        pyro.module("encoder", self.encoder)
        with pyro.plate("cell", sc_count.shape[0]):
            logphi_loc, logphi_scale = self.encoder(sc_count)   # The outputs of encoder become variational paramters.
            logphi = pyro.sample("logphi",dist.Normal(logphi_loc, logphi_scale).to_event(1))
        
        eta = pyro.param("eta", torch.ones(num_topics))   # The parameters stored by pyro.param and above variational parameters will be optimized.
        with pyro.plate("gene", self.gene_size):
            orig_psi = pyro.sample("orig_psi", dist.Multinomial(probs = eta))
        
        mark = (orig_psi != 0)
        psi = torch.zeros_like(orig_psi)
        mean = pyro.param("mean", torch.tensor(0.0))
        std = pyro.param("std", torch.tensor(1.0))
        for col_idx in pyro.plate("topic", num_topics):
            col_mark = mark[:, col_idx]
            num_nonzeros = col_mark.sum().item()
            
            if num_nonzeros>0:
                probs = pyro.sample("probs_{}".format(col_idx), dist.LogNormal(mean, std).expand_by((num_nonzeros,)).to_event(1) )
                norm_probs = F.softmax(probs, dim = -1)
                psi[col_mark, col_idx] = norm_probs
       
        psi = psi.T
        return psi, {"logphi": logphi}


# In[4]:


# Train above model and optimize the variational parameter space in the guide part.
seed = 0
torch.manual_seed(seed)
pyro.set_rng_seed(seed)
num_topics = 75
sc_count = sc_count.float()

pyro.clear_param_store()
vae_LDA = VAE_LDA(
    gene_size=sc_count.shape[1],
    num_topics=num_topics,
    hidden=100,
    dropout=0.2
)
# set up the optimizer
adam_params = {"lr": 0.001, "betas": (0.95, 0.999)}
optimizer = pyro.optim.Adam(adam_params)
# setup the inference algorithm
svi = SVI(vae_LDA.model, vae_LDA.guide, optimizer, loss=Trace_ELBO())

n_steps = 10000
run_loss = np.empty(10000)
# do gradient steps
for step in range(n_steps):
    running_loss = 0.0
    loss = svi.step(sc_count)
    running_loss += loss
    if step % 100 == 0:
        print('.', end='')
    run_loss[step] = running_loss    


# In[5]:


# Visualize the ELBO results along with the steps
# It's expected to decrease to a stable status
import matplotlib.pyplot as plt
x = np.arange(n_steps)
y = np.log(run_loss)
plt.plot(x, y)
plt.xlabel("steps")
plt.ylabel("loss")
plt.savefig("loss_plot.pdf")


# In[6]:


psi = vae_LDA.guide(sc_count)[0].detach().cpu().numpy()    # Extract the optimized psi


# In[7]:


logphi = vae_LDA.guide(sc_count)[1]['logphi']
phi = F.softmax(logphi, -1).detach().cpu().numpy()


# In[8]:


np.savetxt("celda_term_allcells.csv", psi, delimiter = ",")
np.savetxt("celda_loading_allcells.csv", phi, delimiter = ",")



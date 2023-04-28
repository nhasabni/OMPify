import os
import math
import pickle
import json
import logging
from tqdm import tqdm
from pqdm.threads import pqdm
from statistics import mean, geometric_mean
#import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer

from dataset import get_torch_dataloader
from predict import predict

logger = logging.getLogger(__name__)

# --------------------------------------------------
# config params
# --------------------------------------------------
epochs = 10
batch_size = 128
# adam optimizer params
lr = 1e-1
weight_decay = 1e-8

# --------------------------------------------------
# datasets
# --------------------------------------------------

train_loader = get_torch_dataloader('train', use_positive_examples_only=True,
				    												 tokenizer_max_len=256, batch_size=batch_size,
																		 shuffle=True)
valid_loader = get_torch_dataloader('valid', use_positive_examples_only=False,
				    												 tokenizer_max_len=256, batch_size=batch_size,
																		 shuffle=False)
test_loader = get_torch_dataloader('test', use_positive_examples_only=False,
				    												tokenizer_max_len=256, batch_size=batch_size,
																		shuffle=False)

class OMPAutoEncoder(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 256 ==> 9
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(256, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 9)
		)
		
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# The Sigmoid activation function
		# outputs the value between 0 and 1
		# 9 ==> 256
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(9, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 256),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

# Model Initialization
model = OMPAutoEncoder()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
															lr = lr,
															weight_decay = weight_decay)

print('*' * 100)
print("Performing training")
print('*' * 100)

outputs = []
losses = []
for epoch in range(epochs):
  for step, train_code in enumerate(train_loader):
    # Output of Autoencoder
    reconstructed = model(train_code)
    
    # Calculating the loss function
    loss = loss_function(reconstructed, train_code)
    
    # The gradients are set to zero,
    # the gradient is computed and stored.
    # .step() performs parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
	# Storing the losses in a list for plotting
  print("loss:", loss)
  losses.append(loss)
  outputs.append((epochs, train_code, reconstructed))

print('*' * 100)
print("Performing evaluation")
print('*' * 100)

# Use last loss value as the threshold for comparison
threshold = losses[-1]
print("Threshold:", threshold)
  
# Predict on training set itself first
train_loader_with_neg_eg_also = get_torch_dataloader('train', use_positive_examples_only=False,
						     																			tokenizer_max_len=256, batch_size=32,
																											shuffle=False)
predict(model, 'train_with_all_eg', train_loader_with_neg_eg_also, threshold)

# Predict on validation set first
predict(model, 'validation', valid_loader, threshold)
	
# Then test on test set.
predict(model, 'test', test_loader, threshold)
	


		
  

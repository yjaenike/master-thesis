# Utils
from tqdm import tqdm

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_loss(pred, target, lossfn="rmse"):
    
    # flatten targets in order to match predicitions
    target = target.flatten(start_dim=1)
    
    #TODO: Check if NAN's in target, and mask them out in th eloss calculation
    
    # calc loss
    
    if lossfn=="rmse":
        loss = rmse_loss(pred, target)
        
    elif lossfn=="mse":
        loss = mse_loss(pred, target)
    
    
    
    return loss

def mse_loss(pred, target):
    """" Mean Squared Error loss function """
    loss = F.mse_loss(pred, target)
    
    return loss


def rmse_loss(pred, target):
    """ Root Mean Squared Error loss function """
    loss = torch.sqrt(F.mse_loss(pred, target))
    
    return loss

def train_epoch(model, train_dataloader, optimizer, opt, device):
    
    # Set the model into training mode
    model.train()
    
    # Performance 
    losses = []
    
    desc = '  - (Training)   '
    for data, target in tqdm(train_dataloader, mininterval=2, desc=desc, leave=False, ascii=" ▖▘▝▗▚▞█"): #ascii="░▒█"):
        
        # prepare data
        data = data.to(device)
        target = target.to(device)
        
        # Take the last step of the encoder input sequence to use as a startingpoint for the decoder 
        decoder_input = data[:,-1,:].unsqueeze(dim=1)
        
        # forward pass
        optimizer.zero_grad()
        pred = model(data, decoder_input)
        
        # backward pass
        loss = calc_loss(pred, target, opt["loss_func"])
        loss.backward()
        optimizer.step_and_update_lr()
        
        # Sum up loss
        losses.append(loss)
         
    return losses
    
def print_training_settings(device, train_split, eval_split, test_split):
    
    # total samples
    total_samples = train_split.shape[0] + eval_split.shape[0] + test_split.shape[0]
    print("#-------------------------------------#")
    print("\n\033[95m\033[1mTraining Settings:")
    print("- device: ",device)
    print("- train_size:", round(train_split.shape[0]/total_samples,2),"%")
    print("- eval_size:", round(eval_split.shape[0]/total_samples,2),"%")
    print("- test_size:", round(test_split.shape[0]/total_samples,2),"%")
    print("#-------------------------------------#")
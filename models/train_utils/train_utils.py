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
        
    #elif lossfn=="r2":
    #    loss = r_squared(pred, target)
    
    
    
    return loss

def mse_loss(pred, target):
    """" Mean Squared Error loss function """
    loss = F.mse_loss(pred, target)
    
    return loss


def rmse_loss(pred, target):
    """ Root Mean Squared Error loss function """
    loss = torch.sqrt(F.mse_loss(pred, target))
    
    return loss

def r_squared(pred, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    loss = 1 - ss_res / ss_tot
    
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
    
def print_training_settings(device, train_split, eval_split, test_split, lossfn):
    
    # total samples
    total_samples = train_split.shape[0] + eval_split.shape[0] + test_split.shape[0]
    print("\033[95m\033[1m#-------------------------------------#\033[0m")
    print("\033[4m\033[1mTraining Settings:\033[0m")
    print("- \033[1mdevice:\033[0m",device)
    print("- \033[1mtrain_size:\033[0m {}% {}".format(round(train_split.shape[0]/total_samples,2), train_split.shape))
    print("- \033[1meval_size:\033[0m {}% {}".format(round(eval_split.shape[0]/total_samples,2), eval_split.shape))
    print("- \033[1mtest_size:\033[0m {}% {}".format(round(test_split.shape[0]/total_samples,2), test_split.shape))
    print("- \033[1mloss_fn:\033[0m", lossfn)
    print("\033[95m\033[1m#-------------------------------------#\033[0m")

    
def train_old(model, train_dataloader, validation_data, optimizer, device, opt):
    
    # TODO: Implement wandb connection
    
    # TODO: Implement logging
    epoch_losses = []
    
    def print_performance(header, loss_name, loss, start_time, lr):
        print("  - ({header:12}) | {loss_name}: {loss:3.3f} | lr: {lr:8.5f} | elapse: {elapse:3.3f} min".format(header=header,loss_name=loss_name, loss=loss, elapse=(time.time()-start_time)/60, lr=lr))
    
    #TODO: half precission
    
    # TODO: Training Epoch Loop
    for epoch_i in range(opt["epoch"]):
        print("[Epoch: {:>3}]".format(epoch_i))
        
        # trains one epoch
        start = time.time()
        losses = train_epoch(model,train_dataloader,optimizer,opt,device)
        
        #calculate epoch loss and add to logging
        total_epoch_loss = sum([loss.item() for loss in losses])
        epoch_losses.append(total_epoch_loss)
        
        lr = optimizer._optimizer.param_groups[0]['lr']
        
        print_performance('Training',opt["loss_func"], total_epoch_loss, start, lr)
        
        
        #TODO: write function: eval_epoch() 
        #eval_epoch()
    
    
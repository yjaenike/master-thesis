# Utils
from tqdm import tqdm

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

def calc_loss(pred, target, lossfn="rmse"):
    
    # flatten targets in order to match predicitions
    target = target.flatten(start_dim=1)
    
    #TODO: Check if NAN's in target, and mask them out in the loss calculation
    
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

def validate_epoch(model, validation_dataloader, device, opt):
    model.eval()
    running_val_loss = 0
    val_losses = []
    with torch.no_grad():
        for data, target in validation_dataloader:
                
            data = data.to(device)
            target = target.to(device)
            decoder_input = data[:,-1,:].unsqueeze(dim=1)
            
            pred = model(data, decoder_input)
            
            loss = calc_loss(pred, target, opt["loss_func"])
            running_val_loss += loss.item()
            val_losses.append(loss.item())
            
    
    return running_val_loss, val_losses
    

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

def train(model, train_dataloader, validation_datalaoder, optimizer, scheduler, device, opt, name=None):
    
    if opt["use_wandb"]:
        wandb.init(project="masters-thesis", entity="yannicj", config=opt)
        if name:
            wandb.run.name = name

    for epoch in range(opt["epoch"]):
        
        # Initiate running losses for epoch
        running_train_loss = 0
        train_losses = []
        running_val_loss = 0
        val_losses = []
        
        
        with tqdm(train_dataloader, unit="batch", mininterval=1, leave=True) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            # Train epoch
            for data, target in train_dataloader:
                tepoch.update(1)
                
                # Prepare data
                data = data.to(device)
                target = target.to(device)
                decoder_input = data[:,-1,:].unsqueeze(dim=1)
                
                # Trainig step
                optimizer.zero_grad()
                pred = model(data, decoder_input)

                loss = calc_loss(pred, target, opt["loss_func"])
                running_train_loss += loss.item()
                train_losses.append(loss.item())

                loss.backward()
                optimizer.step()
                
                # Update progress bar
                temp_mean_train_loss = sum(train_losses) / len(train_losses)
                tqdm_loss = "{:10.4f}".format(temp_mean_train_loss)
                tepoch.set_postfix(mean_train_loss=tqdm_loss)
            
            # scheduler step
            scheduler.step()
            
            # Validate epoch
            running_val_loss, val_losses = validate_epoch(model, validation_datalaoder, device, opt)
            
            # Update Progress Bar
            mean_tain_loss = sum(train_losses) / len(train_losses)
            mean_val_loss = sum(val_losses) / len(val_losses)
            tqdm_train_loss = "{:10.4f}".format(mean_tain_loss)
            tqdm_val_losses = "{:10.4f}".format(mean_val_loss)
            
            tepoch.set_postfix({"loss": tqdm_train_loss, "val_loss": tqdm_val_losses})
            tepoch.close()
            
            #print(scheduler._last_lr[-1])
            
            if opt["use_wandb"]:
                wandb.log({"train_loss": mean_tain_loss, "val_loss": mean_val_loss, "lr":scheduler._last_lr[-1]})
                wandb.watch(model)
                
            # save model localy
            
    
    if opt["use_wandb"]:
        wandb.finish()

def print_training_settings(device, train_split, eval_split, test_split, opt):
    
    # total samples
    total_samples = train_split.shape[0] + eval_split.shape[0] + test_split.shape[0]
    print("\033[95m\033[1m#-------------------------------------#\033[0m")
    print("\033[4m\033[1mTraining Settings ({}):\033[0m".format(opt["name"]))
    print("- \033[1mdevice:\033[0m",device)
    print("- \033[1mtrain_size:\033[0m {}% {}".format(round(train_split.shape[0]/total_samples,2), train_split.shape))
    print("- \033[1meval_size:\033[0m {}% {}".format(round(eval_split.shape[0]/total_samples,2), eval_split.shape))
    print("- \033[1mtest_size:\033[0m {}% {}".format(round(test_split.shape[0]/total_samples,2), test_split.shape))
    print("- \033[1mloss_fn:\033[0m", opt["loss_func"])
    print("- \033[1mepoch:\033[0m", opt["epoch"])
    print("- \033[1mbatch_size:\033[0m", opt["batch_size"])
    print("- \033[1mwindow_size:\033[0m", opt["window_size"])
    print("- \033[1mlearning_rate:\033[0m", opt["learning_rate"])
    print("\033[95m\033[1m#-------------------------------------#\033[0m")
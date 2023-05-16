import torch
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0 # intialise at 0
    for data in tqdm(data_loader, total=len(data_loader)): # understand tqdm 
        for k, v in data.items():
            data[k] = v.to(device) # understand this
        optimizer.zero_grad()
        _, _, loss = model(**data) # we would need the _ to do inference
        loss.backward()
        optimizer.step() # The optimizer specifices HOW SGD is done. What is one step() here?
        scheduler.step() # whats a scheduler?
        final_loss += loss.item()
    return final_loss / len(data_loader) # this about why this division. Obviously its an average but
                                            # why is the final loss a summation of losses?
                                            # This relates to micrograd and the precise process of backpropagation.

def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        _, _, loss = model(**data) # we would need the _ to do inference
        final_loss += loss.item()
    return final_loss / len(data_loader)
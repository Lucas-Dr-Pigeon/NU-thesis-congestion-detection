# -*- coding: utf-8 -*-
import instant
import preprocessing as prep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np

def run_NN_online(model,running_mode='train', train_set=None, valid_set=None, test_set=None, 
 	batch_size=1, learning_rate=0.01, n_epochs=10, stop_thr=1e-4, shuffle=True):

    
    device=torch.device('cpu')
    if train_set is not None:
    
        train_set_l = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers = 0)
    if test_set is not None:
 	
        test_set_l = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers = 0)
    if valid_set is not None:
 	
        valid_set_l = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle, num_workers = 0)
 	
    loss, accuracy = {'train': [], 'valid': []}, {'train': [], 'valid': []}

 	
    if running_mode is 'train':
		
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
		
        prev_val_loss = 9999999

		
        for i in range(n_epochs):
            
            
            # print (train_set_l.dataset.x.size)
            
            model, train_loss, train_accuracy = _train(model, train_set_l,optimizer, device)
            
 			
            loss['train'].append(train_loss)
 			
            accuracy['train'].append(train_accuracy)
						
 			# val_loss, val_accu = _test(model, valid_set_l, device)
 			
            if valid_set is not None:
				
                valid_loss, valid_accuracy = _test(model, valid_set_l, device)
				
                loss['valid'].append(valid_loss)
				
                accuracy['valid'].append(valid_accuracy)
                
                print ("epoch #" + str(i) + " train: " + str(  round(train_accuracy.item(), 2) ) + " valid: " + str(  round(valid_accuracy.item(), 2) ) )
                

                if abs(valid_loss - prev_val_loss) < stop_thr:
			
                    break
				
                else:
 					
                    prev_val_loss = valid_loss
                    
        if test_set is not None:
            
            test_loss, test_accuracy = _test(model, test_set_l, device)
            
        
        # print (accuracy)
        # print (np.mean(loss['train']))
        # print (np.mean(loss['valid']))
        return model, loss, accuracy, test_accuracy
    
    if running_mode is "test":
        
        test_loss, test_accuracy = _test(model, test_set_l, device)
        
        return test_loss, test_accuracy
    
    
    


def _train(model,data_loader,optimizer,device=torch.device('cpu')):

    criterion = nn.CrossEntropyLoss()
    data_len = len(data_loader)
    targets = data_loader.dataset.y
    targets = np.asarray(targets)
    total_loss = 0
    total_correct = 0
    Preds = []
    
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device).float()
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        preds = outputs.data.max(1)[1]
        loss = criterion(outputs, labels.long())
        # print (outputs.shape)
        # print (labels.shape)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
        total_correct += (preds == labels).sum().float()
    
    return model, total_loss/data_len, total_correct/targets.shape[0] * 100 


def _test(model, data_loader, device=torch.device('cpu')):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    targets = data_loader.dataset.y
    targets = np.asarray(targets)

    total_loss = 0
    total_correct = 0
    data_len = len(data_loader)
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device).float()
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels.long())
            
            preds = outputs.data.max(1)[1]
            # print (preds)
            total_loss += loss.item()
            total_correct += (preds == labels).sum().float()
    # predictions = np.asarray(predictions)
    # losses = np.asarray(predictions)
    return total_loss/data_len, total_correct/targets.shape[0] * 100 
    
    
  
# -*- coding: utf-8 -*-


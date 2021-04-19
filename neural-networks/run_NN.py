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
from os import system
from sklearn.metrics import confusion_matrix, f1_score
def run_NN(model,running_mode='train', train_set=None, valid_set=None, test_set=None, 
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
				
                valid_loss, valid_accuracy, valid_cm, valid_f1 = _test(model, valid_set_l, device)
				
                loss['valid'].append(valid_loss)
				
                accuracy['valid'].append(valid_accuracy)
                
                # if i % 100 == 0:
                #     print ("epoch #" + str(i) + " train: " + str(round(train_accuracy * 100,2)) + " valid: " + str(round(valid_accuracy,2)))
                print ("epoch #" + str(i) + " train: " + str(round(train_accuracy * 100,2)) + " valid: " + str(round(valid_accuracy,2)))
                _ = system('cls')
                ##### debug here #####
                
				
                '''
                print("Epoch ", i)
				
                print("Training Loss ", train_loss, ". Valid Loss ", valid_loss)
				
                # print("Training Accuracy ", train_accuracy, ". Valid Accuracy ", valid_accuracy)
				#####################
				'''
				
                
                if abs(valid_loss - prev_val_loss) < stop_thr:
                    # prev_val_loss = valid_loss
 					
                    break
				
                else:
 					
                    prev_val_loss = valid_loss
                    
        if test_set is not None:
            
            test_loss, test_accuracy, test_cm, test_f1 = _test(model, test_set_l, device)
            
        
        # print (accuracy)
        # print (np.mean(loss['train']))
        # print (np.mean(loss['valid']))
        return model, loss, accuracy, test_accuracy, test_cm, test_f1
    
    if running_mode is "test":
        
        test_loss, test_accuracy, test_cm, test_f1 = _test(model, test_set_l, device)
        
        return test_loss, test_accuracy, test_cm, test_f1
    
    
    


def _train(model,data_loader,optimizer,device=torch.device('cpu')):

    criterion = nn.CrossEntropyLoss()
    data_len = len(data_loader)
    targets = data_loader.dataset.y
    targets = np.asarray(targets)
    total_loss = 0
    total_correct = 0
    Preds = []
    
    
    ims = data_loader.dataset.x
    labels = data_loader.dataset.y
    optimizer.zero_grad()
    outputs = model(ims)
    preds = outputs.data.max(1)[1]
    # print (preds)
    # print (labels)
    # print  (outputs.squeeze(1))
    # labels = np.reshape(labels, (labels.shape[0], 1))
    # preds = torch.tensor(preds, dtype = torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    # print (outputs)
    # print (outputs.shape)
    # print (labels.shape)
    loss = criterion(outputs, labels )
    
    # print (loss)
    loss.backward()
    optimizer.step()
    # print (loss.item())
    
    
    # for i, (images, labels) in enumerate(data_loader):
    #     images = images.to(device).float()
    #     labels = labels.to(device)
    #     optimizer.zero_grad()
    #     outputs = model(images)
    #     preds = outputs.data.max(1)[1]
    #     Preds.append(preds)
    #     loss = criterion(outputs, labels.long())
    #     # print (outputs.shape)
    #     # print (labels.shape)
    #     loss.backward()
    #     optimizer.step()
    #     total_loss += loss.item() 
    #     total_correct += (preds == labels).sum().float()
    # print (np.sum(preds == targets))
    return model, loss, np.sum(preds.numpy() == targets) / preds.shape[0]


def _test(model, data_loader, device=torch.device('cpu')):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    targets = data_loader.dataset.y
    targets = np.asarray(targets)
    # losses = []
    # predictions = []
    total_loss = 0
    total_correct = 0
    data_len = len(data_loader)
    f1 = []
    with torch.no_grad():
        ims = data_loader.dataset.x
        labels = data_loader.dataset.y
        outputs = model(ims)
        preds = outputs.data.max(1)[1]
        loss = criterion(outputs,  torch.tensor(labels, dtype=torch.long))
        # for i, (images, labels) in enumerate(data_loader):
        #     images = images.to(device).float()
        #     labels = labels.to(device)
        #     outputs = model(images)

        #     loss = criterion(outputs, labels.long())
            
        #     preds = outputs.data.max(1)[1]
        #     # print (preds)
        #     total_loss += loss.item()
        #     total_correct += (preds == labels).sum().float()
    # predictions = np.asarray(predictions)
    # losses = np.asarray(predictions)
    # print (preds)
    return loss, np.sum(preds.numpy() == targets)/targets.shape[0] * 100, confusion_matrix(targets, preds.numpy()), f1_score(labels, preds) 
    
    
  

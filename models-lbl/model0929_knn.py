import numpy as np
import pandas as pd
#import networkx as nx
#import math
#import sys
import matplotlib.pyplot as plt
#import collections
import preprocessing as prep
import torch
import torch.nn as nn
import torch.nn.functional as F
from main_full_lane import get_windows, clean_res
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from run_NN import run_NN
from run_NN_online import run_NN_online
from sklearn.ensemble import RandomForestClassifier
import statistics as stats
import random
from sklearn.linear_model import LinearRegression
from KNN0929 import KNN
def clean_info(info, window_size = 100):
    
    infoo = info
    zones_to_del = []

    for zone in info:
        zone_size = len(list(info[zone].keys()))
        if zone_size < 10:
            zones_to_del.append(zone)
    for zone in zones_to_del:
        del info[zone]
        
    return infoo

def get_flow_speed_density(vtx_dict, window, zone):
    num_vehicle = len(list(vtx_dict.keys()))
    dx_list = []
    dt_list = []
    max_x = -1
    min_x = 98989 
    max_t = -1
    min_t = 98989
    v_list = []
    for vehicle in vtx_dict:
        dx = (vtx_dict[vehicle]["x"][-1] - vtx_dict[vehicle]["x"][0]) /3.28084 /1000
        dt = (vtx_dict[vehicle]["t"][-1] - vtx_dict[vehicle]["t"][0]) * 0.1 /3600
        v_list.append(dx/dt)
        dx_list.append(dx)
        dt_list.append(dt)
    dx_list = np.asarray(dx_list)
    dt_list = np.asarray(dt_list)
    v_list = np.asarray(v_list)
    d_x = np.sum(dx_list)
    d_t = np.sum(dt_list)
    ssd = np.std(v_list)
    if not ssd * 0 == 0:
        ssd = 0
    flow = d_x / ((window[1] - window[0])*0.1/3600  * (zone[1] - zone[0])/3.28084 /1000)
    density = d_t / ((window[1] - window[0])*0.1/3600  * (zone[1] - zone[0])/3.28084 /1000)
    speed = flow/density 
    return flow, speed, density, ssd




def get_multiple_snapshots( data, timespan = [1500, 4500], segment = [250, 1750], window_size = 100, zone_size = 200, mpr = 100 ):
    
    Infos = {}
    wzveh = {}
    zones = [tuple([x0, x0 + zone_size]) for x0 in range(segment[0], segment[1], zone_size)]
    windows = [tuple([t0, t0 + window_size]) for t0 in range(timespan[0], timespan[1], window_size)]
    qvk = {}
    vehicles = np.asarray(list(set(data["Vehicle_ID"])))
    num_vehs = vehicles.shape[0]
    all_vehicles = vehicles[ random.sample( range(0, num_vehs), int ( num_vehs * mpr / 100 )) ] if mpr < 100 else vehicles
    
    for i in range(len(data)):
        print (" Processing trajectory #" + str(i))
        row = data[i:i+1]
        lane = row["Lane_ID"][i]
        x = row["Local_Y"][i]
        t = row["Frame_ID"][i]
        v = row["v_Vel"][i]
        veh = row["Vehicle_ID"][i]
        
        if not windows[0][0] <= t < windows[-1][1] or not zones[0][0] <= x < zones[-1][1]:
            continue
        
        
        # try:
        #     zone = zones[int((x - zones[0][0])/ zone_size)]
        #     window = windows[int((t - windows[0][0])/ window_size)]
        # except IndexError:
        #     continue
        zone = zones[int((x - zones[0][0])/ zone_size)]
        window = windows[int((t - windows[0][0])/ window_size)]
        
        
        if lane not in wzveh.keys():
            wzveh[lane] = {}
        if zone not in wzveh[lane].keys():
            wzveh[lane][zone] = {}
        if window not in wzveh[lane][zone].keys():
            wzveh[lane][zone][window] = {}
        if veh not in wzveh[lane][zone][window].keys():
            wzveh[lane][zone][window][veh] = {"x":np.asarray([x]),"t":np.asarray([t])}
        else:
            wzveh[lane][zone][window][veh]["x"] = np.append( wzveh[lane][zone][window][veh]["x"], np.asarray([x]), axis = 0 )
            wzveh[lane][zone][window][veh]["t"] = np.append( wzveh[lane][zone][window][veh]["t"], np.asarray([t]), axis = 0 )
        
        
        if veh not in all_vehicles:
            continue
        
        
        
        if lane not in Infos.keys():
            Infos[lane] = {}
            # wzveh[lane] = {}
        if zone not in Infos[lane].keys():
            Infos[lane][zone] = {}
            # wzveh[lane][zone] = {}
        if window not in Infos[lane][zone].keys():
            Infos[lane][zone][window] = {}
            # wzveh[lane][zone][window] = {}
        if t not in Infos[lane][zone][window].keys():
            Infos[lane][zone][window][t] = np.zeros(shape = (0, 4))
        
        
        
        arr = np.asarray([[x, t, v, veh]])
        Infos[lane][zone][window][t] = np.append(Infos[lane][zone][window][t], arr, axis = 0)
        
        del zone, window
        
    def clean_res(res_dict):
        new_ress = {}
        for zone in res_dict:
            new_ress[zone] = {}
            for window in res_dict[zone]:
                els = res_dict[zone][window]
                # to remove the "nan" values: 
                if els[1] * 0 == 0:
                    new_ress[zone][window] = els  
            # if new_ress[zone] == {}: 
            #     del new_ress[zone]
        return new_ress

    for lane in wzveh:
        qvk[lane] = {}
        for zone in wzveh[lane]:
            qvk[lane][zone] = {}
            for window in wzveh[lane][zone]:
                qvk[lane][zone][window] = get_flow_speed_density(wzveh[lane][zone][window], window, zone)
        qvk[lane] = clean_res(qvk[lane])
    
    return Infos, qvk, wzveh
            
def get_multiple_infos(Infos, window_size = 100):
    
    IInfos = {}
    for lane in Infos:
        info = Infos[lane]
        if lane not in IInfos.keys():
            IInfos[lane] = {}
        for zone in info:
            if zone not in IInfos[lane].keys():
                IInfos[lane][zone] = {}
            for window in info[zone]:
                trjs = Infos[lane][zone][window]
                arr = np.zeros((100,11), dtype = "float")
                arr[:,0] = np.full(100,1.0)
                arr[:,3] = np.full(100,1.0)
                arr[:,4] = np.full(100,1.0)
                arr[:,7] = np.full(100,1.0)
                arr[:,8] = np.full(100,1.0)
                # arr1 = np.zeros((100,7), dtype = "float")
                arr2 = np.zeros((100,4), dtype = "float")
                for frame in info[zone][window]:
                    t = frame - window[0]
                    arr[t][0] = min ( np.mean(trjs[frame][:,2] ) / 70 * 100, 100) / 100
                    arr[t][1] = min ( ( np.max(trjs[frame][:,2]) - np.min(trjs[frame][:,2]) ) / 25 * 100, 100 ) / 100
                    arr[t][2] = min ( stats.pstdev(trjs[frame][:,2]) / 15 * 100, 100 ) / 100
                    arr[t][3] = min ( ( trjs[frame][np.argmax(trjs[frame][:,2])][2] ) / 70 * 100, 100 ) / 100
                    arr[t][4] = min ( ( trjs[frame][np.argmin(trjs[frame][:,2])][2] ) / 70 * 100, 100 ) / 100
                    arr[t][5] = min ( ( trjs[frame][np.argmax(trjs[frame][:,2])][0] - zone[0] ) / 200 * 100, 100 ) / 100
                    arr[t][6] = min ( ( trjs[frame][np.argmin(trjs[frame][:,2])][0] - zone[0] ) / 200 * 100, 100 ) / 100   
                    arr[t][7] = min ( ( trjs[frame][np.argmax(trjs[frame][:,0])][2] ) / 70 * 100, 100 ) / 100
                    arr[t][8] = min ( ( trjs[frame][np.argmin(trjs[frame][:,0])][2] ) / 70 * 100, 100 ) / 100 
                    arr[t][9] = min ( ( trjs[frame][np.argmax(trjs[frame][:,0])][0] - zone[0] ) / 200 * 100, 100 ) / 100
                    arr[t][10] = min ( ( trjs[frame][np.argmin(trjs[frame][:,0])][0] - zone[0] ) / 200 * 100, 100 ) / 100
                    """ mean speed """
                    v_mean = arr[t][0]    
                    """ x_xmax, upstream location """
                    x_xmax = arr[t][9]
                    """ x_xmin, downstream location """
                    x_xmin = arr[t][10]
                   
                    """ array of x """
                    arrX = ( trjs[frame][:,0] - zone[0] ) / 200
                    arrX = (arrX - x_xmin) / (x_xmax - x_xmin) if x_xmax != x_xmin else np.asarray([0.5])
                    """ array of v """
                    arrV = ( trjs[frame][:,2] ) / 70
                                            
                    """ mean speed """
                    arr2[t][0] = arr[t][0]
                    """ speed range """
                    arr2[t][1] = arr[t][1]
                    """ speed standard deviation """
                    arr2[t][2] = arr[t][2]
                    """ lever factor """
                    try:
                        reg = LinearRegression().fit(arrX.reshape(-1,1), arrV)
                        f = reg.coef_[0]
                    except ValueError:
                        f = 0
                    
                    
                    arr2[t][3] = f
                    
                IInfos[lane][zone][window] = arr2
                
    return IInfos
            
            
def clean_multiple_infos(IInfos, window_size = 100, var = 4):
    
    IIInfos = {}
    lane_to_del = []
    for lane in IInfos:
        if len(list(IInfos[lane].keys())) < 3:
            continue
        IIInfos[lane] = clean_info(IInfos[lane], window_size = 100)
        if len(list(IInfos[lane].keys())) < 3:
            lane_to_del.append(lane)
    for lane in lane_to_del:
        del IIInfos[lane]
    
    arr = np.zeros((100,var), dtype = "float")
    arr[:,0] = np.full(100,1.0)
    
    for lane in IIInfos:
        for zone in IIInfos[lane]:
            window_keys = list(IIInfos[lane][zone].keys())
            windows = [tuple([t0, t0 + window_size]) for t0 in range(window_keys[0][0], window_keys[-1][1], window_size)]
            for window in windows:
                if window not in list(IIInfos[lane][zone].keys()):
                    IIInfos[lane][zone][window] = arr
    
    Unions = {}
    for lane in IIInfos:
        Unions[lane] = []
        
        for zone in IIInfos[lane]:
            windows = list(IIInfos[lane][zone].keys())
            Unions[lane] = list(set(windows + Unions[lane]))
    for lane in IIInfos:
        windows = Unions[lane]
        for zone in IIInfos[lane]:
            for window in windows:
                if window not in list(IIInfos[lane][zone].keys()):
                    IIInfos[lane][zone][window] = arr
                
        
    

    return IIInfos


def load_training_data_k(info, res, history = 3, var = 4, boundary = [45, 70]):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys())
    m = len(windows) - history
    features = np.zeros(shape = ((len(zones) - 1) * m, history * 2, 100, var))
    targets = np.zeros(shape = ((len(zones) - 1) * m))
    c = 0
    
    clas = len(boundary) + 1
    for z in range(0, len(zones) - 1):
        zone = zones[z]
        downstream_zone = zones[z + 1]
        for w in range(history, len(windows)):
            window = windows[w]
            
            
            for h in range(history * 2):
                if 0<= h <history:
                    features[c][h] = info[zone][windows[w - h - 1]]
                elif history<= h <2 * history:
                    features[c][h] = info[downstream_zone][windows[w - h - history - 1]]
        
            try:
                k = res[zone][window][2]
            except KeyError:
                k = 0
            # k =  res[zone][window][2]
            if len(boundary) == 2:
                if k < boundary[0]:
                    targets[c] = 0
                    
                elif k >= boundary[1]:
                    targets[c] = 2
                else:
                    targets[c] = 1
            elif len(boundary) == 1:
                targets[c] = 0 if k < boundary[0] else 1
            # print (targets[c])
            c += 1
            
    return features, targets

def load_training_data_u(info, res, history = 3, var = 4, boundary = [15, 45]):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys())
    m = len(windows) - history
    features = np.zeros(shape = ((len(zones) - 1) * m, history * 2, 100, var))
    targets = np.zeros(shape = ((len(zones) - 1) * m))
    c = 0
    
    clas = len(boundary) + 1
    for z in range(1, len(zones)):
        zone = zones[z]
        prev_zone = zones[z - 1]
        for w in range(history, len(windows)):
            window = windows[w]
            
            
            for h in range(history * 2):
                if 0<= h <history:
                    features[c][h] = info[zone][windows[w - h - 1]]
                elif history<= h <2 * history:
                    features[c][h] = info[prev_zone][windows[w - h - history - 1]]
            u =  res[zone][window][1]
            # try:
            #     k = res[zone][window][2]
            # except:
            #     k = 0
            if len(boundary) == 2:
                if u < boundary[0]:
                    targets[c] = 0
                    
                elif u >= boundary[1]:
                    targets[c] = 2
                else:
                    targets[c] = 1
            elif len(boundary) == 1:
                targets[c] = 0 if u < boundary[0] else 1
            # print (targets[c])
            c += 1
    return features, targets

def load_training_data_uk(info, res, history = 3, var = 4, boundary = None):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys())
    m = len(windows) - history
    features = np.zeros(shape = ((len(zones) - 1) * m, history * 2, 100, var))
    targets = np.zeros(shape = ((len(zones) - 1) * m))
    c = 0
    
    for z in range(1, len(zones)):
        zone = zones[z]
        prev_zone = zones[z - 1]
        for w in range(history, len(windows)):
            window = windows[w]
            
            
            for h in range(history * 2):
                if 0<= h <history:
                    features[c][h] = info[zone][windows[w - h - 1]]
                elif history<= h <2 * history:
                    features[c][h] = info[prev_zone][windows[w - h - history - 1]]
            u = res[zone][window][1]
            k = res[zone][window][2]
            # try:
            #     k = res[zone][window][2]
            # except:
            #     k = 0
            if u > 37.5:
                targets[c] = 0
            elif u <= 37.5 and k >= 55:
                targets[c] = 1
            else:
                targets[c] = 2
            # print (targets[c])
            c += 1
    return features, targets
          
def get_features_300x8(IIInfos, ress, history = 3, var = 4, boundary = [45, 70], mode = "density"):

    Features = np.full((0, history*100, var*2), 0.0, dtype = "float")
    Targets = np.full(0, 0.0, dtype = "float")
    Labels = np.full((0,2), 0.0, dtype = "float")
    for lane in IIInfos:
        info = IIInfos[lane]
        res = ress[lane]
        if mode == "density":
            features, targets, labels = load_training_data_300x8(info, res, history, var, boundary) 
        elif mode == "speed":
            features, targets = load_training_data_u(info, res, history, var, boundary)
        elif mode == "dual":
            features, targets = load_training_data_uk(info, res, history, var)
        Features = np.append(Features, features, axis = 0)
        Targets = np.append(Targets, targets, axis = 0)
        Labels = np.append(Labels, labels, axis = 0)
        # print (np.where(Targets == 1))
    return Features, Targets, Labels
        
        
    
    
    
    # for lane in np.arange(1, 9):
    #     _sn, iinfo = get_snapshots(data, timespan, segment, window_size, zone_size, lane)
    #     Infos[lane] = clean_info(iinfo)
    #     del iinfo
    # for lane in Infos:
    #     iinfo = Infos[lane]
        
        
    
    





    
    
    
    

def get_a_input(info, pred_zone, pred_window, window_num = 4):
    
    first_zone = list(info.keys())[0]
    first_window = list(info[first_zone].keys())[0]
    window_size = first_window[1] - first_window[0]
    shape = info[first_zone][first_window].shape
    input = np.zeros(shape=(0, shape[0], shape[1])) 
    ws = np.arange(pred_window[0] - window_num * window_size, pred_window[0], window_size)
    
    for zone in info:
        if zone != pred_zone:
            continue
        for w in ws:
            arr = np.asarray([info[zone][tuple([w, w+window_size])]])
            input = np.append(input, arr, axis = 0)
                
    return np.asarray([input])
 


def plot_speed_range_time_series(info, res, timespan = [1000, 4000] , zone_id = 0 ):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys()) 
    zone = zones[zone_id]
    densities = np.zeros(0)
    speed_ranges = np.zeros(0)
    
    for window in windows:
        speed_ranges = np.append( speed_ranges, info[zone][window][:,1] )
        densities = np.append( densities, np.full(100, res[zone][window][2]) )
        # densities.append(np.full(100, res[zone][window][2]))
    N = len(speed_ranges)
    plt.ylim(0, 100)
    plt.xlim(0, 3000)
    plt.scatter(np.arange(N), speed_ranges, color = "red", s = 0.1)
    plt.scatter(np.arange(N), densities, color = "blue", s = 0.1)
    plt.show()
    
def plot_max_v_loc_time_series(info, res, timespan = [1000, 4000] , zone_id = 0 ):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys()) 
    zone = zones[zone_id]
    densities = np.zeros(0)
    speed_ranges = np.zeros(0)
    
    for window in windows:
        speed_ranges = np.append( speed_ranges, info[zone][window][:,2] )
        densities = np.append( densities, np.full(100, res[zone][window][2]) )
        # densities.append(np.full(100, res[zone][window][2]))
    N = len(speed_ranges)
    plt.ylim(0, 100)
    plt.xlim(0, 3000)
    plt.scatter(np.arange(N), speed_ranges, color = "green", s = 0.1)
    plt.scatter(np.arange(N), densities, color = "blue", s = 0.1)
    plt.show()
    
def plot_min_v_loc_time_series(info, res, timespan = [1000, 4000] , zone_id = 0 ):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys()) 
    zone = zones[zone_id]
    densities = np.zeros(0)
    speed_ranges = np.zeros(0)
    
    for window in windows:
        speed_ranges = np.append( speed_ranges, info[zone][window][:,3] )
        densities = np.append( densities, np.full(100, res[zone][window][2]) )
        # densities.append(np.full(100, res[zone][window][2]))
    N = len(speed_ranges)
    plt.ylim(0, 100)
    plt.xlim(0, 3000)
    plt.scatter(np.arange(N), speed_ranges, color = "orange", s = 0.1)
    plt.scatter(np.arange(N), densities, color = "blue", s = 0.1)
    plt.show()
            
def data_split(features, targets, ratio = 0.8):
    
    N = features.shape[0]
    N_train = int (N * ratio)
    N_valid = int ((N - N_train) / 2)
    N_test = N - N_train - N_valid
    iterations = int(1 / (1 - ratio))
    train_ids = np.zeros((iterations, N_train))
    validtest_ids = np.zeros((iterations, N_valid + N_test))
    valid_ids = np.zeros((iterations, N_valid))
    test_ids = np.zeros((iterations, N_test))
    arr = np.arange(0, N)
    for itr in range(train_ids.shape[0] - 1):
        validtest_ids[itr] = np.arange(N - (N_valid + N_test) * (itr + 1), N - itr * (N_valid + N_test) )
        train_ids[itr] = np.setdiff1d(arr, validtest_ids[itr])
        valid_ids[itr] = np.arange(N - (N_valid + N_test) * (itr + 1), N - (N_valid + N_test) * (itr + 1) + N_valid )
        test_ids[itr] = np.setdiff1d(validtest_ids[itr], valid_ids[itr])
        
        
    validtest_ids[-1] = np.arange(0, (N_valid + N_test))
    train_ids[-1] = np.setdiff1d(arr, validtest_ids[-1])
    valid_ids[-1] = np.arange(0, N_valid)
    test_ids[-1] = np.setdiff1d(validtest_ids[-1], valid_ids[-1])
    return train_ids.astype(int), valid_ids.astype(int), test_ids.astype(int)


class CNN(nn.Module):
    
    def __init__(self, kernel_size, stride):
        super(CNN, self).__init__()    
        self.conv1 = nn.Conv2d(7,32,kernel_size[0],stride=stride[0])
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(32,64,kernel_size[1],stride=stride[1])
        self.fc1 = nn.Linear(3136,128)
        self.fc2 = nn.Linear(128,2)
        # self.fc3 = nn.Linear(32,4)
        # self.fc4 = nn.Linear(4,2)
        
    def forward(self, input):
        
        output = torch.tensor(input, dtype=torch.float)
        # print (output.shape)
        output = output.permute((0,3,1,2))
        # print (output.shape)
        output = self.pool(F.relu(self.conv1(output))) 
        # print (output.shape)
        # output = self.pool(F.relu(self.conv2(output))) 
        output = output.view(output.size(0),-1)     
        output = self.fc1(output)
        output = self.fc2(output)
        # output = self.fc3(output)
        # output = self.fc4(output)
        return output

class NN(nn.Module):
    
    def __init__(self):
        super(NN, self).__init__()    
        # self.fc1 = nn.Linear(4200,1024)
        # self.fc2 = nn.Linear(1024,128)
        # self.fc3 = nn.Linear(128,32)
        # self.fc4 = nn.Linear(32,8)
        # self.fc5 = nn.Linear(8,2)
        
        self.fc1 = nn.Linear(2400, 32)
        self.fc2 = nn.Linear(32, 3)
        
    def forward(self, input):
        
        output = torch.tensor(input, dtype=torch.float)
        # output = output.permute((0,3,1,2))
        output = output.view(output.size(0),-1)
        output = self.fc1(output)
        output = self.fc2(output)
        
        # output = self.fc3(output)
        # output = self.fc4(output)
        # output = self.fc5(output)
        return output

    
			
		
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return (x, y)

    def __len__(self):
        return self.x.shape[0]


def cross_validation(features, targets, rate = 0.8, just_one_trial = False, n_epochs=1*10**4,stop_thr=1e-7, mode = "offline"):
    order = np.arange(features.shape[0])
    np.random.shuffle(order)
    features = features[order]
    targets = targets[order]
        
    print (features.shape)
    train_ids, valid_ids, test_ids = data_split(features, targets, rate)
    accuracies = []
    all_accuracy = []
    test_accuracies = []
    for i in range( int(1/(1-rate)) ): 
        trainset = MyDataset(features[train_ids[i]], targets[train_ids[i]])
        validset = MyDataset(features[valid_ids[i]], targets[valid_ids[i]])
        testset = MyDataset(features[test_ids[i]], targets[test_ids[i]])
        if mode == "offline":
            _model,_loss,_accuracy,_test_accuracy = run_NN(RNN(),running_mode='train', train_set=trainset, valid_set=validset, test_set=testset, 
                                                           batch_size=1, learning_rate=0.01, n_epochs=n_epochs, stop_thr=stop_thr, shuffle=True)
        elif mode == "online":
            _model,_loss,_accuracy,_test_accuracy = run_NN_online(RNN(),running_mode='train', train_set=trainset, valid_set=validset, test_set=testset, 
                                                           batch_size=1, learning_rate=0.01, n_epochs=n_epochs, stop_thr=stop_thr, shuffle=True)
        accuracies.append(_accuracy["valid"][-1])
        all_accuracy.append([_accuracy["train"][-1], _accuracy["valid"][-1]])
        test_accuracies.append(_test_accuracy)
        
        if just_one_trial:
            break
        
    accuracies = np.asarray(accuracies)
    final_model = _model
    # print (accuracies)
    print ("Final Accuracy: " + str(np.mean(accuracies)))
    return accuracies, _model, all_accuracy, test_accuracies
 
def get_features_from_single_dataset(path,timespan = [1000,5000], segment = [1000, 2400], window_size = 100, zone_size = 200, mpr = 100, boundary = [40, 60], mode = "density", history = 3):
    path = r"vehicle-trajectory-data\0805am-0820am\trajectories-0805am-0820am.csv"
    us101_data = pd.read_csv(path)
    us101 = prep.dataset(data = us101_data, vehicles = [])
    Infos, qvk, wzveh = get_multiple_snapshots(us101_data, timespan, segment, window_size, zone_size, mpr)

    IInfos = get_multiple_infos(Infos)
    IIInfos = clean_multiple_infos(IInfos, var = 4)
    features, targets, labels = get_features_300x8(IIInfos, qvk, history, var = 4, boundary = boundary, mode = mode)
    return features, targets

def Generate_Features(Paths, timespan, segment, mpr, boundary, window_size=100, zone_size=200, mode="density", var=4, history = 3, one_iter = False):
    Features = np.full((0, history * 100, var * 2), 0.0, dtype = "float")
    Targets = np.full(0, 0.0, dtype = "float")
    all_qvks = {}
    Labells = np.full((0,2), 0.0, dtype = "float")
    for path in Paths:
        data = pd.read_csv(path)
        Infos, qvk, wzveh = get_multiple_snapshots(data, timespan, segment, window_size, zone_size, mpr)
        IInfos = get_multiple_infos(Infos)
        IIInfos = clean_multiple_infos(IInfos)
        features, targets, labells = get_features_300x8(IIInfos, qvk, history, var, boundary)
        Features = np.append(Features, features, axis = 0)
        Targets = np.append(Targets, targets, axis = 0)
        all_qvks[path] = qvk
        if one_iter:
            break
        Labells = np.append(Labells, labells, axis = 0)
    return Features, Targets, Labells
        
    


def plot_q_k(qvk, thresholds = [60]):
        if len(thresholds) == 1:
            Q0 = []
            K0 = []
            Q1 = []
            K1 = []
            for lane in qvk:
                for zone in qvk[lane]:
                    for window in qvk[lane][zone]:
                        if qvk[lane][zone][window][2] < thresholds[0]:
                            Q0.append(qvk[lane][zone][window][0])
                            K0.append(qvk[lane][zone][window][2])
                        else:
                            Q1.append(qvk[lane][zone][window][0])
                            K1.append(qvk[lane][zone][window][2])
            plt.scatter(K0, Q0, color = "blue", s = 0.1)
            plt.scatter(K1, Q1, color = "red", s = 0.1)
            plt.ylim(0, 3000)
            plt.xlim(0, 140)
            plt.show()
        elif len(thresholds) == 2:
            Q0 = []
            K0 = []
            Q1 = []
            K1 = []
            Q2 = []
            K2 = []
            for lane in qvk:
                for zone in qvk[lane]:
                    for window in qvk[lane][zone]:
                        if qvk[lane][zone][window][2] < thresholds[0]:
                            Q0.append(qvk[lane][zone][window][0])
                            K0.append(qvk[lane][zone][window][2])
                        elif qvk[lane][zone][window][2] >= thresholds[1]:
                            Q2.append(qvk[lane][zone][window][0])
                            K2.append(qvk[lane][zone][window][2])
                        else:
                            Q1.append(qvk[lane][zone][window][0])
                            K1.append(qvk[lane][zone][window][2])
            plt.scatter(K0, Q0, color = "blue", s = 0.1)
            plt.scatter(K2, Q2, color = "red", s = 0.1)
            plt.scatter(K1, Q1, color = "orange", s = 0.1)
            plt.ylim(0, 3000)
            plt.xlim(0, 140)
            plt.show() 
    
def plot_q_k_by_uk_criteraion(qvk):
    Q0 = []
    K0 = []
    Q1 = []
    K1 = []
    Q2 = []
    K2 = []
    for dataset in qvk:
        for lane in qvk[dataset]:
            for zone in qvk[dataset][lane]:
                for window in qvk[dataset][lane][zone]:
                    q, u, k = [qvk[dataset][lane][zone][window][0], qvk[dataset][lane][zone][window][1], qvk[dataset][lane][zone][window][2]]
                    if u > 37.5:
                        Q0.append(q)
                        K0.append(k)
                    elif u <= 37.5 and k > 55:
                        Q2.append(q)
                        K2.append(k)
                    else:
                        Q1.append(q)
                        K1.append(k)
    plt.scatter(K0, Q0, color = "blue", s = 0.1)
    plt.scatter(K2, Q2, color = "red", s = 0.1)
    plt.scatter(K1, Q1, color = "orange", s = 0.1)
    plt.ylim(0, 3000)
    plt.xlim(0, 140)
    plt.show() 
    
def plot_q_k_by_4clusters(qvk):
    Q0 = []
    K0 = []
    Q1 = []
    K1 = []
    Q2 = []
    K2 = []
    Q3 = []
    K3 = []
    for dataset in qvk:
        for lane in qvk[dataset]:
            for zone in qvk[dataset][lane]:
                for window in qvk[dataset][lane][zone]:
                    q, u, k = [qvk[dataset][lane][zone][window][0], qvk[dataset][lane][zone][window][1], qvk[dataset][lane][zone][window][2]]
                    if u > 45 and k <= 40:
                        Q0.append(q)
                        K0.append(k)
                    elif u <= 45 and k <= 40:
                        Q2.append(q)
                        K2.append(k)
                    elif u > 45 and k > 40:
                        Q1.append(q)
                        K1.append(k)
                    elif u <= 45 and k > 40:
                        Q3.append(q)
                        K3.append(k)
                    
    plt.scatter(K0, Q0, color = "green", s = 0.1)
    plt.scatter(K2, Q2, color = "orange", s = 0.1)
    plt.scatter(K1, Q1, color = "blue", s = 0.1)
    plt.scatter(K3, Q3, color = "red", s = 0.1)
    plt.ylim(0, 3000)
    plt.xlim(0, 140)
    plt.show() 
    
def supplement_features(paths, Features, Targets, timespan = [900, 4900], segment = [900, 2300], keep = [1, 2], times = 1, mpr = 100, boundary = [40, 60], mode = "density", history = 3, window_size = 100, zone_size = 200):
    for t in range(times):
        path = paths[t]
        data = pd.read_csv(path)
        Infos, qvk, wzveh = get_multiple_snapshots(data, timespan, segment, window_size, zone_size, mpr)
        IInfos = get_multiple_infos(Infos)
        IIInfos = clean_multiple_infos(IInfos)
        features, targets, _ = get_features_300x8(IIInfos, qvk, history, var=4, boundary=boundary)
        # idx = np.where(targets == ke)[0]
        # features1 = features[idx]
        # targets1 = targets[idx]
        Features = np.append(Features, features, axis = 0)
        Targets = np.append(Targets, targets, axis = 0) 
        # if len(keep):
        #     for ke in keep:
        #         idx = np.where(targets == ke)[0]
        #         features1 = features[idx]
        #         targets1 = targets[idx]
        #         Features = np.append(Features, features1, axis = 0)
        #         Targets = np.append(Targets, targets1, axis = 0) 
    return Features, Targets
        
def save_data(data, filepath):
    import pickle
    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    
def load_data(filepath):
    import pickle
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)
    return data

# def plot_lever_factor_vs_density(Info, QVK, lane, zone):
#     fs = []
#     ks = []
#     trjs = Info[lane][zone]
#     stats = QVK[lane][zone]
#     for window in trjs:
#         k = stats[window]
#         for t in range(trjs[window].shape[0]):
#             fs.append(trjs[window][t][3])
#             ks.append(stats[window][2] / 120)
#     fs = np.asarray(fs)
#     ks = np.asarray(ks)
#     plt.scatter(np.arange(fs.shape[0]), fs, s=0.1, color = "black")
#     plt.scatter(np.arange(ks.shape[0]), ks, s=0.1, color = "red")
#     plt.show()

def plot_lever_factor_vs_density(Info, QVK, lane, zone, timespan):
    fs = []
    ks = []
    qs = []
    trjs = Info[lane][zone]
    stats = QVK[lane][zone]
    windows = [tuple([w, w+100]) for w in np.arange(timespan[0], timespan[1], 100)]
    for window in windows:
        arr = trjs[window]
        for t in range(arr.shape[0]):
            fs.append(arr[t][3])
        ks.append(stats[window][2])   
        qs.append(stats[window][0])  
    fs = np.asarray(fs)
    plt.plot(np.arange(fs.shape[0]), fs, color = "black")
    
    plt.plot(np.arange(fs.shape[0]), np.zeros(fs.shape[0]), linestyle = "dashed", color = "red")
    plt.xlim(0, 400)
    plt.ylim(-0.25, 0.25)
    plt.xlabel("time frame (0.1sec)")
    plt.ylabel("slope")
    plt.title("slopes of u-x distribution over 400 time steps")
    plt.show()
    
    plt.scatter(ks, qs, c = "red", s = 5)
    plt.xlim(0, 120)
    plt.ylim(0, 3000)
    
    Ks = []
    Qs = []
    for lane in QVK:
        for zone in QVK[lane]:
            for window in QVK[lane][zone]:
                Ks.append(QVK[lane][zone][window][2])
                Qs.append(QVK[lane][zone][window][0])
    Ks = np.asarray(Ks)
    Qs = np.asarray(Qs)
    
    plt.scatter(Ks, Qs, s = 0.5, color = "grey")
    plt.xlabel("Density veh/km")
    plt.ylabel("Flow veh/km")
    plt.title("Q-K plot in a sample")
    plt.show()
    fs = []
    ks = []
    windows = list(trjs.keys())
    steps = [ ss[0]+50 for ss in windows ]
    frames = np.arange(windows[0][0], windows[-1][1])
    for window in trjs:
        arr = trjs[window]
        for t in range(arr.shape[0]):
            fs.append(arr[t][3])
        ks.append(stats[window][2])   
    fs = np.asarray(fs)
    
    plt.scatter(steps, ks, color = "blue")
    plt.plot(steps, ks, color = "black", linestyle = "dashed")
    plt.xlabel("Frames (0.1sec)")
    plt.ylabel("Density (veh/km)")
    plt.ylim(0, 120)
    plt.xlim(1000, 5000)
    plt.title("Density changes over time")
    plt.show()
    plt.plot(frames, fs, linewidth = 0.2)
    plt.plot(frames, np.zeros(len(frames)), linestyle ="dashed", color = "red")
    plt.xlabel("Frames (0.1sec)")
    plt.ylabel("Slope")
    plt.ylim(-0.3, 0.3)
    plt.xlim(1000, 5000)
    plt.title("Fluctuation of u-x distribution function slope over time")
    plt.show()
    
class RNN(nn.Module):
    
    def __init__(self):
        super(RNN, self).__init__()    
        self.hidden_dim = 2
        self.layer_dim = 2
        self.input_dim = 8
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.layer_dim, batch_first = True)
        self.fc = nn.Linear(self.hidden_dim, 2)
        
    def forward(self, x):
        x = torch.Tensor(x)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        out, _ = self.rnn(x, h0)
        print (out.shape)
        output = self.fc(out) 
        
        print (output.shape)
        return output
    
def load_training_data_300x8(info, res, history = 3, var = 4, boundary = [45, 70]):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys())
    m = len(windows) - history
    features = np.zeros(shape = ((len(zones) - 1) * m, history * 100, var * 2))
    targets = np.zeros(shape = ((len(zones) - 1) * m))
    labels = np.zeros(shape = ((len(zones) - 1) * m, 2))
    c = 0
    
    clas = len(boundary) + 1
    for z in range(0, len(zones) - 1):
        zone = zones[z]
        downstream_zone = zones[z + 1]
        for w in range(history, len(windows)):
            window = windows[w]
            
            
            for h in range(history):
                # print (features[c][h*100:(h+1)*100, 0:var].shape)
                features[c][h*100:(h+1)*100, 0:var] = info[zone][windows[w - h - 1]]
                features[c][h*100:(h+1)*100, var:var*2] = info[zone][windows[w - h - 1]]
                
        
            try:
                targets[c] = res[zone][window][1]
            except KeyError:
                targets[c] = 0

            labels[c][0] = zone[0] + 100
            labels[c][1] = window[0] + 50
            c += 1
    return features, targets, labels

def plot_KNN_results(testTargets, Predictions, testLabels):
    plt.scatter(testLabels, Predictions, color = "red", label = "Predicted Speed")
    plt.scatter(testLabels, testTargets, color = "blue", label = "Actual Speed")
    plt.plot(testLabels, Predictions, color = "black", linestyle = "dashed")
    plt.plot(testLabels, testTargets, color = "black", linestyle = "dashed")
    plt.ylim(0,80)
    plt.xlabel("Time frame (0.1 sec)")
    plt.ylabel("Speed (km/hr)")
    plt.title("Predicted Speed vs Actual Speed")
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    
    paths = [r"vehicle-trajectory-data\0820am-0835am\trajectories-0820am-0835am.csv", 
             r"vehicle-trajectory-data\0750am-0805am\trajectories-0750am-0805am.csv", 
             r"vehicle-trajectory-data\0805am-0820am\trajectories-0805am-0820am.csv" 
              ]
    
    # Features100_300x8, Targets100_300x8, _ = Generate_Features(paths[:2],[1000,5000],[1000,2400],100,[40,60],one_iter = False,history=3)
    
    # Features100_300x8, Targets100_300x8 = supplement_features(paths[:2], Features100_300x8, Targets100_300x8, timespan = [1000, 5000], segment = [900, 2300], times = 2, mpr = 50)
    # Features100_300x8, Targets100_300x8 = supplement_features(paths[:2], Features100_300x8, Targets100_300x8, timespan = [1000, 5000], segment = [950, 2350], times = 2, mpr = 50)
    # Features100_300x8, Targets100_300x8 = supplement_features(paths[:2], Features100_300x8, Targets100_300x8, timespan = [1000, 5000], segment = [1050, 2450], times = 2, mpr = 50)
    
    # testFeatures, testTargets, testLabels = Generate_Features([paths[2]],[1000,5000],[1000,2400],50,[40,60],one_iter = False,history=3)
    # testFeatures100, testTargets100, testLabels100 = Generate_Features([paths[2]],[1000,5000],[1000,2400],100,[40,60],one_iter = False,history=3)
    # save_data(Features100_300x8, "./data/Features100_300x8_continuous")
    # save_data(Targets100_300x8, "./data/Targets100_300x8_continuous")
    
    # Features100_300x8 = load_data("./data/Features100_300x8_continuous")
    # Targets100_300x8 = load_data("./data/Targets100_300x8_continuous")
    
    # Features50_300x8 = Features100_300x8
    # Targets50_300x8 = Targets100_300x8 
 
    
    # kEstimator = KNN(Features50_300x8, Targets50_300x8)
    # outcomes_speed = kEstimator.__predict__(testFeatures[888:924], testTargets[888:924],variables_to_take = [0,3] ,k = 20)
    
    # plot_KNN_results(testTargets[888:924], outcomes_speed[0][:37], testLabels[:,1][888:924])
    
    # print (outcomes[2])
    
    
    # Features100_k60s = load_data("./data/Features100_k60_s")
    # Targets100_k60s = load_data("./data/Targets100_k60_s")
    
    # acc100_k60s, model100_k60s, accuracy100_k60s, test_accuracy100_k60s = cross_validation(Features100_k60s, Targets100_k60s, rate = 0.8, just_one_trial = 
    #                                                                                                     False, n_epochs=1*10**4, stop_thr=1e-6, mode = "online")
    

    # acc100_k4060s, model100_k4060s, accuracy100_k4060s, test_accuracy100_k4060s = cross_validation(Features100_k4060s, Targets100_k4060s, rate = 0.8, just_one_trial = 
    #                                                                                                     False, n_epochs=1*10**4, stop_thr=1e-6)


    # Features100_k4060, Targets100_k4060, _ = Generate_Features(paths,[1000,5000],[1000,2400],100,[40,60],one_iter = False,history=3)
    # Features100_k4060s, Targets100_k4060s = supplement_features(paths, Features100_k4060s, Targets100_k4060s, timespan = [920, 4920], segment = [970, 2370], keep = [2], times = 3, mpr = 100, boundary = [40, 60], mode = "density")
    # print (np.sum(Targets100_k4060s == 0), np.sum(Targets100_k4060s == 1), np.sum(Targets100_k4060s == 2))
    # acc100_k4060s, model100_k4060s, accuracy100_k4060s, test_accuracy100_k4060s = cross_validation(Features100_k4060s, Targets100_k4060s, rate = 0.8, just_one_trial = 
    #                                                                                                     False, n_epochs=1*10**4, stop_thr=1e-6)
    # save_data(Features100_k4060, "Features100_k4060_0927")
    # save_data(Targets100_k4060, "Targets100_k4060_0927")
    
    # path = r"vehicle-trajectory-data\0820am-0835am\trajectories-0820am-0835am.csv"
    # us101_data = pd.read_csv(path)
    # Infos100, qvk100, wzveh100 = get_multiple_snapshots(us101_data, [1000,5000], segment = [1000, 2400], window_size = 100, zone_size = 200, mpr = 100)

    # IInfos100 = get_multiple_infos(Infos100)
    # IIInfos100 = clean_multiple_infos(IInfos100)
    # features, targets = get_features_300x8(IIInfos100, qvk100, history = 3, var = 4)
    # plot_lever_factor_vs_density(IIInfos100, qvk100, 2, tuple([1600, 1800]), [1000,1400])
    
    # train_ids, valid_ids, test_ids = data_split(features, targets)
    
    # trainset = MyDataset(features[train_ids[0]], targets[train_ids[0]])
    # validset = MyDataset(features[valid_ids[0]], targets[valid_ids[0]])
    # testset = MyDataset(features[test_ids[0]], targets[valid_ids[0]])
    # _,_,accuracy4,test_accuracy4 = run_NN(NN(),running_mode='train', train_set=trainset, valid_set=validset, test_set=validset, 
    #     batch_size=1, learning_rate=0.01, n_epochs=10000, stop_thr=1e-6, shuffle=True)# -*- coding: utf-8 -*-


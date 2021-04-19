import numpy as np
import pandas as pd
#import networkx as nx
import math
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
        if dt > 0:
            v_list.append(dx/dt)
        dx_list.append(dx)
        dt_list.append(dt)
    dx_list = np.asarray(dx_list)
    dt_list = np.asarray(dt_list)
    v_list = np.asarray(v_list)
    d_x = np.sum(dx_list)
    d_t = np.sum(dt_list)
    ssd = np.std(v_list)
    ran = np.max(v_list) - np.min(v_list) if v_list.shape[0] else 0
    if not ssd * 0 == 0:
        ssd = 0
    flow = d_x / ((window[1] - window[0])*0.1/3600  * (zone[1] - zone[0])/3.28084 /1000)
    density = d_t / ((window[1] - window[0])*0.1/3600  * (zone[1] - zone[0])/3.28084 /1000)
    speed = np.mean(v_list) if not math.isnan(np.mean(v_list)) else 0
    ran = ran if not math.isnan(ran) else 0
    return flow, speed, density, ssd, ran




def get_multiple_snapshots_segments( data, timespan = [1500, 4500], segments = [[800,2400],[850,2450],[900,2500],[950,2550],[1000,2600]], window_size = 100, zone_size = 200, mpr = 100 ):
    
    Infos = {}
    wzveh = {}
    qvk = {}
    all_sections = []
    for segment in segments:
        all_sections = [tuple([x0, x0 + zone_size]) for x0 in range(segment[0], segment[1], zone_size)] + all_sections
        
    windows = [tuple([t0, t0 + window_size]) for t0 in range(timespan[0], timespan[1], window_size)]
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
        
        
        if not windows[0][0] <= t < windows[-1][1]:
            continue
        
        
        for zone in all_sections:
         
            
            window = windows[int((t - windows[0][0])/ window_size)]
            if not zone[0] <= x < zone[1]:
                continue
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
                
            if zone not in Infos[lane].keys():
                Infos[lane][zone] = {}  
            if window not in Infos[lane][zone].keys():
                Infos[lane][zone][window] = {}
            if veh not in Infos[lane][zone][window].keys():
                Infos[lane][zone][window][veh] = {}
                Infos[lane][zone][window][veh]["x"] = []
                Infos[lane][zone][window][veh]["t"] = []
 
            
            Infos[lane][zone][window][veh]["x"].append(x) 
            Infos[lane][zone][window][veh]["t"].append(t) 
 
            
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
    # Ks = []
    # Qs = []
    IInfos = {}
    for lane in Infos:
        info = Infos[lane]
        if lane not in IInfos.keys():
            IInfos[lane] = {}
        for zone in info:
            zone_size = zone[1] - zone[0]
            if zone not in IInfos[lane].keys():
                IInfos[lane][zone] = {}
            for window in info[zone]:
                res = get_flow_speed_density(Infos[lane][zone][window], window, zone)
                
                # Ks.append(res[2])
                # Qs.append(res[0])
                # plt.scatter(Ks, Qs, s = 0.1, color = "grey")
                IInfos[lane][zone][window] = np.zeros(2)
                # IInfos[lane][zone][window][0] = 1 if math.isnan(min (res[1] / 65, 1)) else min (res[1] / 65, 1)
                IInfos[lane][zone][window][0] = min (res[1] / 65, 1)
                try:
                    IInfos[lane][zone][window][1] = min (res[3] / 10, 1)
                except ZeroDivisionError:
                    IInfos[lane][zone][window][1] = 0
                
    return IInfos
            
            
def clean_multiple_infos(IInfos, window_size = 100, var = 2):
    
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
    
    arr = np.zeros(2, dtype = "float")

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


def load_training_data_k(info, res, history = 3, var = 2, boundary = [45, 70], zone_size = 200, freeflows = True):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys())
    m = len(windows) - history
    features = np.zeros(shape = ((len(zones) - 1) * m, history * 2 * var))
    targets = np.zeros(shape = ((len(zones) - 1) * m))
    c = 0

    for zone in info:
        downstream_zone = tuple([zone[0] + zone_size, zone[1] + zone_size])
        if zone not in zones or downstream_zone not in zones:
            continue
        for w in range(history, len(windows)):
            window = windows[w]
            if window not in list(res[zone].keys()):
                continue
                
            if freeflows == False and res[zone][window][2] < 30:
                continue
            
            cc = 0
            for h in range(history * 2):
                
                if 0<= h <history:
                    for v in range(var):
                        features[c][cc] = info[zone][windows[w - h - 1]][v]
                        cc += 1
                        
                elif history<= h <2 * history:
                    for v in range(var):
                        features[c][cc] = info[downstream_zone][windows[w - h - history - 1]][v]
                        cc += 1
            try:
                k = res[zone][window][2]
            except KeyError:
                k = 0
            
            if len(boundary) == 2:
                if k < boundary[0]:
                    targets[c] = 0
                    
                elif k >= boundary[1]:
                    targets[c] = 2
                else:
                    targets[c] = 1
            elif len(boundary) == 1:
                targets[c] = 0 if k < boundary[0] else 1
            # print (features[c])
            # print (targets[c])
            c += 1
            
    return features, targets

          
def get_features(IIInfos, ress, history = 3, var = 2, boundary = [45, 70], mode = "density", freeflows = True):

    Features = np.full((0, history * 2 * var), 0.0, dtype = "float")
    Targets = np.full(0, 0.0, dtype = "float")
    for lane in IIInfos:
        info = IIInfos[lane]
        res = ress[lane]
        features, targets = load_training_data_k(info, res, history, var, boundary, freeflows = freeflows) 
        Features = np.append(Features, features, axis = 0)
        Targets = np.append(Targets, targets, axis = 0)
        # print (np.where(Targets == 1))
    return Features, Targets
        
        
    
  




    
    
    
    

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
        
        self.fc1 = nn.Linear(4, 2)
        # self.fc2 = nn.Linear(4, 2)
        # self.fc2 = nn.Linear(3, 2)
        
    def forward(self, input):
        
        # print (input.shape)
        output = torch.tensor(input, dtype=torch.float)
        # print (output.shape)
        # output = output.permute((0,3,1,2))
        output = output.view(output.size(0),-1)
        output = self.fc1(output)
        # output = self.fc2(output)
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



def load_three_datasets_features(paths, timespan=[1000,5000], segment=[1000,2400], window_size=100, zone_size=200, history=3, var=2, mpr = 100, mode = "density", boundary = [50]):
    
    Features = np.full((0, history * 2, 100, var), 0.0, dtype = "float")
    Targets = np.full(0, 0.0, dtype = "float")
    all_qvks = {}
    for path in paths:
        us101_data = pd.read_csv(path)
        us101 = prep.dataset(data = us101_data, vehicles = [])
        Infos, qvk, all_vehicles = get_multiple_snapshots(us101_data, timespan, segment, window_size, zone_size, mpr)
        IInfos = get_multiple_infos(Infos)
        IIInfos = clean_multiple_infos(IInfos, var)
        features, targets = get_features(IIInfos, qvk, history = history, var = var, boundary = boundary, mode = mode)
        # features, targets = get_features(IIInfos, qvk, history = 3, var = 7, boundary = [19,37.5], mode = "speed")
        # features, targets = get_features(IIInfos, qvk, history = 3, var = 7, boundary = [], mode = "dual")
        Features = np.append(Features, features, axis = 0)
        Targets = np.append(Targets, targets, axis = 0)
        all_qvks[path] = qvk
    return Features, Targets, all_qvks

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
    test_cms = []
    for i in range( int(1/(1-rate)) ): 
        trainset = MyDataset(features[train_ids[i]], targets[train_ids[i]])
        validset = MyDataset(features[valid_ids[i]], targets[valid_ids[i]])
        testset = MyDataset(features[test_ids[i]], targets[test_ids[i]])
        if mode == "offline":
            _model,_loss,_accuracy,_test_accuracy, _test_cm = run_NN(NN(),running_mode='train', train_set=trainset, valid_set=validset, test_set=testset, 
                                                           batch_size=1, learning_rate=0.01, n_epochs=n_epochs, stop_thr=stop_thr, shuffle=True)
        elif mode == "online":
            _model,_loss,_accuracy,_test_accuracy, _test_cm = run_NN_online(NN(),running_mode='train', train_set=trainset, valid_set=validset, test_set=testset, 
                                                           batch_size=1, learning_rate=0.01, n_epochs=n_epochs, stop_thr=stop_thr, shuffle=True)
        accuracies.append(_accuracy["valid"][-1])
        all_accuracy.append([_accuracy["train"][-1], _accuracy["valid"][-1]])
        test_accuracies.append(_test_accuracy)
        test_cms.append(_test_cm)
        
        if just_one_trial:
            break
        
    accuracies = np.asarray(accuracies)
    final_model = _model
    # print (accuracies)
    print ("Final Accuracy: " + str(np.mean(accuracies)))
    return accuracies, _model, all_accuracy, test_accuracies, test_cms
 
def get_features_from_single_dataset(path,timespan = [1000,5000], segment = [1000, 2400], window_size = 100, zone_size = 200, mpr = 100, boundary = [40, 60], mode = "density", history = 3):
    path = r"vehicle-trajectory-data\0805am-0820am\trajectories-0805am-0820am.csv"
    us101_data = pd.read_csv(path)
    us101 = prep.dataset(data = us101_data, vehicles = [])
    Infos, qvk, wzveh = get_multiple_snapshots(us101_data, timespan, segment, window_size, zone_size, mpr)

    IInfos = get_multiple_infos(Infos)
    IIInfos = clean_multiple_infos(IInfos, var = 2)
    features, targets = get_features(IIInfos, qvk, history, var = 2, boundary = boundary, mode = mode)
    return features, targets

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
        features, targets = get_features(IIInfos, qvk, history, var=2, boundary=boundary)
        for ke in keep:
            idx = np.where(targets == ke)[0]
            features1 = features[idx]
            targets1 = targets[idx]
            Features = np.append(Features, features1, axis = 0)
            Targets = np.append(Targets, targets1, axis = 0) 
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
def _smooth_(y, box_pts=100):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_lever_factor_vs_density(Info, QVK, lane, zone, timespan, smooth = False):
    fs = []
    ks = []
    qs = []
    trjs = Info[lane][zone]
    stats = QVK[lane][zone]
    # windows = [tuple([w, w+100]) for w in np.arange(timespan[0], timespan[1], 100)]
    # for window in windows:
    #     fs.append(Info[lane][zone][window][1])
    #     ks.append(stats[window][2])   
    #     qs.append(stats[window][0])  
    # fs = np.asarray(fs)
    # plt.plot(np.arange(fs.shape[0]), fs, color = "black")
    
    # plt.plot(np.arange(fs.shape[0]), np.zeros(fs.shape[0]), linestyle = "dashed", color = "red")
    # plt.xlim(0, 400)
    # plt.ylim(-0.25, 0.25)
    # plt.xlabel("time frame (0.1sec)")
    # plt.ylabel("slope")
    # plt.title("slopes of u-x distribution over 400 time steps")
    # plt.show()
    
    # # plt.scatter(ks, qs, c = "red", s = 5)
    # plt.xlim(0, 120)
    # plt.ylim(0, 3000)
    
    # Ks = []
    # Qs = []
    # for lane in QVK:
    #     for zone in QVK[lane]:
    #         for window in QVK[lane][zone]:
    #             Ks.append(QVK[lane][zone][window][2])
    #             Qs.append(QVK[lane][zone][window][0])
    # Ks = np.asarray(Ks)
    # Qs = np.asarray(Qs)
    
    # plt.scatter(Ks, Qs, s = 0.5, color = "grey")
    # plt.xlabel("Density veh/km")
    # plt.ylabel("Flow veh/km")
    # plt.title("Flow-Density Relationship")
    # plt.show()
    fs = []
    ks = []
    us = []
    windows = list(trjs.keys())
    steps = [ ss[0]+50 for ss in windows ]
    frames = np.arange(windows[0][0], windows[-1][1])
    for window in trjs:
        arr = trjs[window]
        fs.append(arr[1])
        ks.append(stats[window][2])   
        us.append(stats[window][1])   
    fs = np.asarray(fs)
    us = np.asarray(us)
    fig, ax = plt.subplots()
    ax.scatter(steps, ks, color = "blue")
    ax.plot(steps, ks, color = "black", linestyle = "dashed")
    ax.set_xlabel("Frames (0.1sec)")
    ax.set_ylabel("Density (veh/km)", color = "blue")
    ax.set_ylim(0, 120)
    ax.set_xlim(1000, 5000)
    
    ax2 = ax.twinx()
    if smooth:
        fs = _smooth_(fs)
    # print (fs2 == fs.all())
    ax2.plot(steps, fs, linewidth = 0.2, color = "red")
    ax2.plot(steps, np.zeros(len(steps)), linestyle ="dashed", color = "grey")
    ax2.set_ylabel("Slope", color = "red")
    ax2.set_ylim(-1, 1)
    # plt.xlim(1000, 5000)
    plt.title("u-x slope changes vs actual density changes")
    plt.show()
    
    # plt.scatter(steps, us, color = "green")
    # plt.plot(steps, us, color = "black", linestyle = "dashed")
    # plt.ylim(0, 70)
    
    # plt.show()
    
def Supplement_Features_01(paths, Features, Targets, timespan = [900, 4900], segments = [[1000, 2400]], times = 1, mpr = 100, boundary = [40, 60], mode = "density", history = 3, window_size = 100, zone_size = 200, freeflows = True):
    
    for t in range(times):
        if np.sum(Targets == 0) >= np.sum(Targets == 1):
            keep = [1]
        else:
            keep = [0,1]
        
        path = paths[t]
        data = pd.read_csv(path)
        Infos, qvk, wzveh, Infos2 = get_multiple_snapshots_segments(data, timespan, segments, window_size, zone_size, mpr)
        IInfos = get_multiple_infos(Infos, Infos2)
        IIInfos = clean_multiple_infos(IInfos)
        features, targets = get_features(IIInfos, qvk, history, var=2, boundary=boundary, mode=mode, freeflows=freeflows)
        # print (features.shape)
        for ke in keep:
            idx = np.where(targets == ke)[0]
            features1 = features[idx]
            targets1 = targets[idx]
            Features = np.append(Features, features1, axis = 0)
            Targets = np.append(Targets, targets1, axis = 0) 
    return Features, Targets


def Generate_Features(Paths, timespan, segments, mpr, boundary, window_size=100, zone_size=200, mode="density", var=2, history = 3, one_iter = False, freeflows = True):
    Features = np.full((0, history * 2 * var), 0.0, dtype = "float")
    Targets = np.full(0, 0.0, dtype = "float")
    all_qvks = {}
    for path in Paths:
        data = pd.read_csv(path)
        Infos, qvk, wzveh = get_multiple_snapshots_segments(data, timespan, segments, window_size, zone_size, mpr)
        IInfos = get_multiple_infos(Infos)
        IIInfos = clean_multiple_infos(IInfos)
        features, targets = get_features(IIInfos, qvk, history, var, boundary, mode, freeflows)
        Features = np.append(Features, features, axis = 0)
        Targets = np.append(Targets, targets, axis = 0)
        all_qvks[path] = qvk
        if one_iter:
            break
    return Features, Targets, IIInfos

def Drop_Features(Features, Targets):
    N0 = np.sum(Targets == 0)
    N1 = np.sum(Targets == 1)
    Nd = int (N0 - N1)
    idx0 = np.where(Targets == 0)[0]
    idx0_to_drop = random.sample( list(idx0), Nd )
    New_Features = np.delete(Features, idx0_to_drop, axis = 0)
    New_Targets = np.delete(Targets, idx0_to_drop, axis = 0)
    return New_Features, New_Targets
 
def confusion_matrix(Predictions, Actuals):
    els = list(set(Actuals))
    ''' pred : actual '''
    cm = np.zeros(shape = (len(els), len(els))) 
    
    for i, pred in enumerate(Predictions):
        actual = Actuals[i]
        cm[int(pred)][int(actual)] += 1
    return cm
    
   
def Cross_Validation_Random_Forest(features, targets, rate = 0.8):
    order = np.arange(features.shape[0])
    np.random.shuffle(order)
    features = features[order]
    targets = targets[order]
        
    train_ids, valid_ids, test_ids = data_split(features, targets, rate)
    accuracies = []
    all_accuracy = []
    test_accuracies = []
    test_cms = []
    for i in range( int(1/(1-rate)) ): 
        train_Features = features[train_ids[i]]
        train_Targets = targets[train_ids[i]]
        test_Features = np.append(features[valid_ids[i]], features[test_ids[i]], axis = 0)
        test_Targets = np.append(targets[valid_ids[i]], targets[test_ids[i]], axis = 0)
        rf = RandomForestClassifier(max_depth=2, random_state=0)
        rf.fit(train_Features, train_Targets)
        rfpreds = rf.predict(test_Features)
        rfcm = confusion_matrix(rfpreds, test_Targets)
        rfacc = round ((rfcm[0][0] + rfcm[1][1]) / (rfcm[0][0] + rfcm[1][1] + rfcm[0][1] + rfcm[1][0]) * 100, 2)
        test_cms.append(rfcm)
        test_accuracies.append(rfacc)
    return test_cms, np.mean(np.asarray(test_accuracies))
    
def Extract_nonZero_Features(features, targets, var = 2, history = 1):
    idx_to_keep = []
    for f in range(features.shape[0]):
        if np.all(features[f] == 0) or np.isnan(features[f]).any():
            continue
        else:
            idx_to_keep.append(f)
    del f      
    idx_to_keep = np.asarray(idx_to_keep)
    newfeatures = features[idx_to_keep]
    newtargets = targets[idx_to_keep]
    return newfeatures, newtargets

def Permute(features):
    N = features.shape[0]
    M = features.shape[1]
    arrs = np.zeros(shape = [N, 1, M])
    for n in range(N):
        for m in range(M):
            arrs[n][0][m] = features[n][m]
    print (arrs.shape)
    return arrs

def Plot_States(IIInfos, qvk):
    Ks = []
    Qs = []
    for lane in IIInfos:
        for zone in IIInfos[lane]:
            for window in IIInfos[lane][zone]:
                try:
                    res = qvk[lane][zone][window]
                    Ks.append(res[2])
                    Qs.append(res[0])
                except KeyError:
                    pass
                
    plt.scatter(Ks, Qs, s=0.1, color="grey")
    plt.show()
                
def Cross_Validation_SVM(features, targets, rate = 0.8):
    order = np.arange(features.shape[0])
    np.random.shuffle(order)
    features = features[order]
    targets = targets[order]
        
    train_ids, valid_ids, test_ids = data_split(features, targets, rate)
    accuracies = []
    all_accuracy = []
    test_accuracies = []
    test_cms = []
    for i in range( int(1/(1-rate)) ): 
        train_Features = features[train_ids[i]]
        train_Targets = targets[train_ids[i]]
        test_Features = np.append(features[valid_ids[i]], features[test_ids[i]], axis = 0)
        test_Targets = np.append(targets[valid_ids[i]], targets[test_ids[i]], axis = 0)
        machine = svm.SVC()
        machine.fit(train_Features, train_Targets)
        svmpreds = machine.predict(test_Features)
        svmcm = confusion_matrix(svmpreds, test_Targets)
        svmacc = round ((svmcm[0][0] + svmcm[1][1]) / (svmcm[0][0] + svmcm[1][1] + svmcm[0][1] + svmcm[1][0]) * 100, 2)
        test_cms.append(svmcm)
        test_accuracies.append(svmacc)
    return test_cms, np.mean(np.asarray(test_accuracies))
if __name__ == "__main__":
    
    paths = [r"vehicle-trajectory-data\0820am-0835am\trajectories-0820am-0835am.csv", 
             r"vehicle-trajectory-data\0750am-0805am\trajectories-0750am-0805am.csv", 
             r"vehicle-trajectory-data\0805am-0820am\trajectories-0805am-0820am.csv" 
              ]

    # segments = [[0,2000],[50,1850],[100,1900],[150,1950],[25,1825],[75,1875],[125,1925],[175,1975]]
    
    segments = [[800,2000],[850,1850],[900,1900],[950,1950],[825,1825],[875,1875],[925,1925],[975,1975]]
   
    ''' 
    Run the following scripts to get the result in 100% connectivity
    '''
    # Features100_k60z, Targets100_k60z, IIInfos100_k60z = Generate_Features(paths,[1000,5000],segments,100,[60],one_iter = False,history=1,zone_size=200)
    # Features100_k60z, Targets100_k60z = Supplement_Features_01(paths, Features100_k60z, Targets100_k60z, timespan = [1000, 5000], segments = segments_to_supplement, times = 3, mpr = 100, boundary = [60], mode = "density", history = 1, window_size = 100, zone_size = 200)
    # NewFeatures100_k60z, NewTargets100_k60z = Extract_nonZero_Features(Features100_k60z,Targets100_k60z)
    # NewFeatures100_k60z, NewTargets100_k60z = Drop_Features(NewFeatures100_k60z, NewTargets100_k60z)
    # outcomesZ = Cross_Validation_Random_Forest(NewFeatures100_k60z, NewTargets100_k60z, rate = 0.8)
    # acc100k60z, model100k60z, accuracy100k60z, test_accuracy100k60z, cms100k60z = cross_validation(NewFeatures100_k60z, NewTargets100_k60z, rate = 0.8, just_one_trial = False, n_epochs=3*10**5, stop_thr=1e-12, mode = "offline")
    
    # save_data(Features100_k60z, "Features100k60z")
    # save_data(Targets100_k60z, "Targets100k60z")
    '''
    Run the following scripts to get the result in 50% connectivity
    '''
    # Features50k60e, Targets50k60e, IIInfos50k60e = Generate_Features(paths,[1000,5000],segments,50,[60],one_iter = False,history=1,zone_size=200, freeflows = True)
    
    # NewFeatures50k60e, NewTargets50k60e = Extract_nonZero_Features(Features50k60e,Targets50k60e)
    # NewFeatures50k60e, NewTargets50k60e = Drop_Features(NewFeatures50k60e, NewTargets50k60e)
    
    # save_data(Features50k60e, "Features50k60e")
    # save_data(Targets50k60e, "Targets50k60e")
    
    
    # Features50k60e, Targets50k60e = [ load_data("Features50k60e"), load_data("Targets50k60e")  ]
    
    outcomesSVM50k60e = Cross_Validation_Random_Forest(NewFeatures50k60e, NewTargets50k60e, rate = 0.8)
    outcomesRF50k60e = Cross_Validation_Random_Forest(NewFeatures50k60e, NewTargets50k60e, rate = 0.8)
    # outcomesNN50k60e = cross_validation(NewFeatures50k60e, NewTargets50k60e, rate = 0.8, just_one_trial = False, n_epochs=3*10**4, stop_thr=1e-12, mode = "offline")
    

    # trainset = MyDataset(NewFeatures50k60z, NewTargets50k60z)
    # validset = MyDataset(NewFeatures50k60z, NewTargets50k60z)
    # testset = MyDataset(NewFeatures50k60z, NewTargets50k60z)
    # results50k60z = run_NN(model50k60z,running_mode='test', train_set=trainset, valid_set=validset, test_set=testset, 
    #                                                        batch_size=1, learning_rate=0.01, n_epochs=1*10**5, stop_thr=1e-6, shuffle=True)
    
    '''
    Run the following scripts to get the result in 30% connectivity
    '''
    
    # Features30k60z, Targets30k60z, IIInfos30k60z = Generate_Features(paths,[1000,5000],segments,30,[60],one_iter = False,history=1,zone_size=200)
    # Features30k60z, Targets30k60z = Supplement_Features_01(paths, Features30k60z, Targets30k60z, timespan = [1000, 5000], segments = segments_to_supplement, times = 3, mpr = 30, boundary = [60], mode = "density", history = 1, window_size = 100, zone_size = 200)
    # NewFeatures30k60z, NewTargets30k60z = Extract_nonZero_Features(Features30k60z,Targets30k60z)
    # NewFeatures30k60z, NewTargets30k60z = Drop_Features(NewFeatures30k60z, NewTargets30k60z)
    # outcomesRF30k60z = Cross_Validation_Random_Forest(NewFeatures30k60z, NewTargets30k60z, rate = 0.8)
    # acc30k60z, model30k60z, accuracy30k60z, test_accuracy30k60z, cms30k60z = cross_validation(NewFeatures30k60z, NewTargets30k60z, rate = 0.8, just_one_trial = False, n_epochs=3*10**5, stop_thr=1e-12, mode = "offline")
    
    # save_data(Features30k60z, "Features30k60z")
    # save_data(Targets30k60z, "Targets30k60z")
    
    # Features30k60z = load_data("Features30k60z")
    # Targets30k60z = load_data("Targets30k60z")
    
    '''
    Run the following scripts to get the result with density 50veh/km as threshold in 50% connectivity
    '''
    
    # Features50k50z, Targets50k50z, IIInfos50k50z = Generate_Features(paths,[1000,5000],segments,50,[50],one_iter = False,history=1,zone_size=200)
    # Features50k50z, Targets50k50z = Supplement_Features_01(paths, Features50k50z, Targets50k50z, timespan = [1000, 5000], segments = segments_to_supplement, times = 3, mpr = 50, boundary = [50], mode = "density", history = 1, window_size = 100, zone_size = 200)
    # NewFeatures50k50z, NewTargets50k50z = Extract_nonZero_Features(Features50k50z,Targets50k50z)
    # NewFeatures50k50z, NewTargets50k50z = Drop_Features(NewFeatures50k50z, NewTargets50k50z)
    # outcomes50Z = Cross_Validation_Random_Forest(NewFeatures50k50z, NewTargets50k50z, rate = 0.8)
    # acc50k50z, model50k50z, accuracy50k50z, test_accuracy50k50z, cms50k50z = cross_validation(NewFeatures50k50z, NewTargets50k50z, rate = 0.8, just_one_trial = False, n_epochs=3*10**5, stop_thr=1e-6, mode = "online")
    # save_data(Features50k50z, "Features50k50z")
    # save_data(Targets50k50z, "Targets50k50z")
    
    
    
    
    
    # print (np.sum(Targets100_k60z == 0), np.sum(Targets100_k60z == 1))
    
    
    # Features100_k60e, Targets100_k60e, IIInfos100_k60e = Generate_Features(paths,[1000,5000],segments,100,[60],one_iter = False,history=1,zone_size=200)
    # Features100_k60e, Targets100_k60e = Supplement_Features_01(paths, Features100_k60e, Targets100_k60e, timespan = [1000, 5000], segments = segments_to_supplement, times = 3, mpr = 100, boundary = [60], mode = "density", history = 1, window_size = 100, zone_size = 200)
    # NewFeatures100_k60e, NewTargets100_k60e = Extract_nonZero_Features(Features100_k60e,Targets100_k60e)
    # NewFeatures100_k60e, NewTargets100_k60e = Drop_Features(NewFeatures100_k60e, NewTargets100_k60e)
    # outcomes = Cross_Validation_Random_Forest(NewFeatures100_k60e, NewTargets100_k60e, rate = 0.8)
    # acc100k60e, model100k60e, accuracy100k60e, test_accuracy100k60e, cms100k60e = cross_validation(NewFeatures100_k60e, NewTargets100_k60e, rate = 0.8, just_one_trial = False, n_epochs=3*10**5, stop_thr=1e-6, mode = "offline")
    
    # print (np.sum(Targets100_k60e == 0), np.sum(Targets100_k60e == 1))
    

    # save_data(Features50_k50, "Features50_k50_Elfars")
    # save_data(Targets50_k50, "Targets50_k50_Elfars")
    
   
    '''
    Run the following scripts to get the result in 50% connectivity with 2 history
    '''
    # Features50k60z2, Targets50k60z2, IIInfos50k60z2 = Generate_Features(paths,[1000,5000],segments,50,[60],one_iter = False,history=2,zone_size=200, freeflows = True)
    # 
    # NewFeatures50k60z2, NewTargets50k60z2 = Extract_nonZero_Features(Features50k60z2,Targets50k60z2)
    # NewFeatures50k60z2, NewTargets50k60z2 = Drop_Features(NewFeatures50k60z2, NewTargets50k60z2)
    
    # save_data(Features50k60z2, "Features50k60z2")
    # save_data(Targets50k60z2, "Targets50k60z2")
    
    # 
    # Features50k60z2, Targets50k60z2 = [load_data("Features50k60z2"), load_data("Targets50k60z2") ]
    
    # outcomesRF50k60Z2 = Cross_Validation_Random_Forest(NewFeatures50k60z2, NewTargets50k60z2, rate = 0.8)
    # outcomes50k60Z2 = cross_validation(NewFeatures50k60z2, NewTargets50k60z2, rate = 0.8, just_one_trial = False, n_epochs=3*10**5, stop_thr=1e-6, mode = "offline")
    
    
    # Features50k60z, Targets50k60z = [load_data("Features50k60z"), load_data("Targets50k60z") ]
    # NewFeatures50k60z, NewTargets50k60z = Extract_nonZero_Features(Features50k60z,Targets50k60z)
    # NewFeatures50k60z, NewTargets50k60z = Drop_Features(NewFeatures50k60z, NewTargets50k60z)
    
    # outcomes50k60Z = cross_validation(NewFeatures50k60z, NewTargets50k60z, rate = 0.8, just_one_trial = False, n_epochs=3*10**5, stop_thr=1e-6, mode = "offline")
    '''
    See the intermediate variables of the model
    '''
    # us101_data = pd.read_csv(paths[0])
    # Infos100h, qvk100h, wzveh100h = get_multiple_snapshots_segments(us101_data, [1000,5000], segments = segments, window_size = 100, zone_size = 200, mpr = 100)
    # IInfos100h = get_multiple_infos(Infos100h)
    # IIInfos100h = clean_multiple_infos(IInfos100h)
    # featuresh, targetsh = get_features(IIInfos100h, qvk100h, history = 1, var = 2, boundary = [60])
    # featuresh, targetsh =  Extract_nonZero_Features(featuresh, targetsh)
    # Plot_States(IIInfos100h, qvk100h)
    # plot_lever_factor_vs_density(IIInfos100h, qvk100h, 1, tuple([800,1000]), [1000, 5000], smooth = False)
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


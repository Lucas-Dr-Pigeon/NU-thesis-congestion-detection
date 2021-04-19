import numpy as np
import pandas as pd
#import networkx as nx
#import math
#import sys
import matplotlib.pyplot as plt
#import collections
import preprocessing as prep
import math
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
from sklearn.preprocessing import PolynomialFeatures
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
                arr2 = np.zeros((100,2), dtype = "float")
                for frame in info[zone][window]:
                    t = frame - window[0]
                    arr[t][0] = min ( np.mean(trjs[frame][:,2] ) / 70 * 100, 100) / 100 if trjs[frame][:,2].shape[0] else 1 
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
                    arrX = ( trjs[frame][:,0] - zone[0] ) / 100
                    arrX = (arrX - x_xmin) / (x_xmax - x_xmin) if x_xmax != x_xmin else np.asarray([0.5])
                    
                    """ array of v """
                    arrV = ( trjs[frame][:,2] ) / 70
                    arrV = np.sqrt(arrV)
                    """ mean speed """
                    arr2[t][0] = arr[t][0]
                    
                    """ speed standard deviation with sign"""
                    """ to get the sign """
                    try:
                        reg = LinearRegression().fit(arrX.reshape(-1,1), arrV)
                        f = reg.coef_[0] 
                    except ValueError:
                        f = 0
                    if t > 0:
                        _f = prev_arr2[1]
                        arr2[t][1] = (0.5*_f + 0.5*f)
                    else:
                        arr2[t][1] = f
                    # arr2[t][1] = f / abs(f) * arr[t][1]
                    arr2[t][1] = 0 if math.isnan(arr2[t][1]) else arr2[t][1]
                    prev_arr2 = arr2[t]
                IInfos[lane][zone][window] = arr2
                
                
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


def load_training_data_k(info, res, history = 3, var = 2, boundary = [45, 70]):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys())
    m = len(windows) - history
    features = np.zeros(shape = ((len(zones)) * m, history, 100, var))
    targets = np.zeros(shape = ((len(zones)) * m))
    c = 0
    
    clas = len(boundary) + 1
    for z in range(0, len(zones)):
        zone = zones[z]
        for w in range(history, len(windows)):
            window = windows[w]
            
            
            for h in range(history):
                features[c][h] = info[zone][window]
                
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


          
def get_features(IIInfos, ress, history = 3, var = 2, boundary = [45, 70], mode = "density"):

    Features = np.full((0, history, 100, var), 0.0, dtype = "float")
    Targets = np.full(0, 0.0, dtype = "float")
    for lane in IIInfos:
        info = IIInfos[lane]
        res = ress[lane]
        if mode == "density":
            features, targets = load_training_data_k(info, res, history, var, boundary) 
        elif mode == "speed":
            features, targets = load_training_data_u(info, res, history, var, boundary)
        elif mode == "dual":
            features, targets = load_training_data_uk(info, res, history, var)
        Features = np.append(Features, features, axis = 0)
        Targets = np.append(Targets, targets, axis = 0)
        # print (np.where(Targets == 1))
    return Features, Targets
        
        
    
    
    
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
        
        self.fc1 = nn.Linear(600, 200)
        self.fc2 = nn.Linear(200, 32)
        self.fc3 = nn.Linear(32, 2)
        
    def forward(self, input):
        
        output = torch.tensor(input, dtype=torch.float)
        # output = output.permute((0,3,1,2))
        output = output.view(output.size(0),-1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
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

def cross_validation(features, targets, rate = 0.8, just_one_trial = False, n_epochs=1*10**5,stop_thr=1e-12, mode = "offline"):
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
    test_f1 = []
    for i in range( int(1/(1-rate)) ): 
        trainset = MyDataset(features[train_ids[i]], targets[train_ids[i]])
        validset = MyDataset(features[valid_ids[i]], targets[valid_ids[i]])
        testset = MyDataset(features[test_ids[i]], targets[test_ids[i]])
        if mode == "offline":
            _model,_loss,_accuracy,_test_accuracy,_test_cm, _test_f1 = run_NN(NN(),running_mode='train', train_set=trainset, valid_set=validset, test_set=testset, 
                                                           batch_size=1, learning_rate=0.01, n_epochs=n_epochs, stop_thr=stop_thr, shuffle=True)
        elif mode == "online":
            _model,_loss,_accuracy,_test_accuracy,_test_c, _test_f1 = run_NN_online(NN(),running_mode='train', train_set=trainset, valid_set=validset, test_set=testset, 
                                                           batch_size=1, learning_rate=0.01, n_epochs=n_epochs, stop_thr=stop_thr, shuffle=True)
        accuracies.append(_accuracy["valid"][-1])
        all_accuracy.append([_accuracy["train"][-1], _accuracy["valid"][-1]])
        test_accuracies.append(_test_accuracy)
        test_cms.append(_test_cm)
        test_f1.append(_test_f1)
        if just_one_trial:
            break
        
    accuracies = np.asarray(accuracies)
    final_model = _model
    # print (accuracies)
    print ("Final Accuracy: " + str(np.mean(accuracies)))
    return accuracies, _model, all_accuracy, test_accuracies, test_cms, round(np.mean(np.asarray(test_f1)) * 100, 2)
 
def get_features_from_single_dataset(path,timespan = [1000,5000], segment = [1000, 2400], window_size = 100, zone_size = 200, mpr = 100, boundary = [40, 60], mode = "density", history = 3):
    path = r"vehicle-trajectory-data\0805am-0820am\trajectories-0805am-0820am.csv"
    us101_data = pd.read_csv(path)
    us101 = prep.dataset(data = us101_data, vehicles = [])
    Infos, qvk, wzveh = get_multiple_snapshots(us101_data, timespan, segment, window_size, zone_size, mpr)

    IInfos = get_multiple_infos(Infos)
    IIInfos = clean_multiple_infos(IInfos, var = 2)
    features, targets = get_features(IIInfos, qvk, history, var = 2, boundary = boundary, mode = mode)
    return features, targets

def Generate_Features(Paths, timespan, segment, mpr, boundary, window_size=100, zone_size=200, mode="density", var=2, history = 3, one_iter = False):
    Features = np.full((0, history, 100, var), 0.0, dtype = "float")
    Targets = np.full(0, 0.0, dtype = "float")
    all_qvks = {}
    for path in Paths:
        data = pd.read_csv(path)
        Infos, qvk, wzveh = get_multiple_snapshots(data, timespan, segment, window_size, zone_size, mpr)
        IInfos = get_multiple_infos(Infos)
        IIInfos = clean_multiple_infos(IInfos)
        features, targets = get_features(IIInfos, qvk, history, var, boundary)
        Features = np.append(Features, features, axis = 0)
        Targets = np.append(Targets, targets, axis = 0)
        all_qvks[path] = qvk
        if one_iter:
            break
    return Features, Targets, all_qvks
    
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
def _smooth_(y, box_pts=50):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_lever_factor_vs_density(Info, QVK, lane, zone, timespan, smooth = False, mpr = 100):
    fs = []
    ks = []
    qs = []
    trjs = Info[lane][zone]
    stats = QVK[lane][zone]
    windows = [tuple([w, w+100]) for w in np.arange(timespan[0], timespan[1], 100)]
    for window in windows:
        arr = trjs[window]
        for t in range(arr.shape[0]):
            fs.append(arr[t][1])
        ks.append(stats[window][2])   
        qs.append(stats[window][0])  
    fs = np.asarray(fs)
    
    # plt.plot(np.arange(fs.shape[0]), fs, color = "black")
    
    # plt.plot(np.arange(fs.shape[0]), np.zeros(fs.shape[0]), linestyle = "dashed", color = "red")
    # plt.xlim(0, 400)
    # plt.ylim(-0.25, 0.25)
    # plt.xlabel("time frame (0.1sec)")
    # plt.ylabel("slope")
    # plt.title("slopes of u-x distribution over 400 time steps")
    # plt.show()
    
    # plt.scatter(ks, qs, c = "red", s = 5)
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
    # plt.title("Q-K plot in a sample")
    # plt.show()
    plt.style.use('seaborn-whitegrid')
    
    
    fs = []
    ks = []
    windows = list(trjs.keys())
    steps = [ ss[0]+50 for ss in windows ]
    frames = np.arange(windows[0][0], windows[-1][1])
    for window in windows:
        arr = trjs[window]
        for t in range(arr.shape[0]):
            fs.append(arr[t][1])
        ks.append(stats[window][2])   
    fs = np.asarray(fs)
    if smooth:
        fs = _smooth_(fs)
    sort = sorted(zip(steps, ks))
    tuples = zip(*sort)
    steps, ks = [list(tu) for tu in tuples]
    fig, ax = plt.subplots()
    ax.scatter(steps, ks, color = "blue", zorder = 3)
    ax.plot(steps, ks, color = "black", linestyle = "dashed", zorder = 2)
    ax.set_xlabel("Frames (0.1sec)")
    ax.set_ylabel("Density (veh/km)", color = "blue")
    ax.set_ylim(0, 120)
    ax.set_xlim(1000, 5000)
    # plt.title("Density changes over time")
    # plt.show()
    ax2 = ax.twinx()
    ax2.plot(np.arange(timespan[0], timespan[1]), fs, linewidth = 0.3, color = "red", zorder = 1)
    ax2.plot(np.arange(timespan[0], timespan[1]), np.zeros(len(np.arange(fs.shape[0]))), linewidth = 0.5, linestyle ="dashed", color = "black", zorder = 1)
    ax2.set_ylabel("Speed Distribution Coefficient", color = "red")
    ax2.set_ylim(-0.3, 0.3 )
    plt.savefig("SDC"+str(lane)+str(zone)+str(mpr), dpi=300)
    # plt.xlim(1000, 5000)
    # plt.title("Fluctuation of u-x distribution function slope over time")
    plt.show()
       
def Clear_Features(features, targets):
    idx = np.arange(features.shape[0])
    idx_to_keep = []
    for i in idx:
        zeroi = 0
        
        for j in range(features[i].shape[0]):
            zerocount = 0
            for k in range(features[i][j].shape[0]):
                # print (features[i][j][k])
                if np.all(features[i][j][k] == 0):
                    zerocount += 1
            if zerocount > 50:
                zeroi += 1
                break
        if zeroi:
            continue
        else:
            idx_to_keep.append(i)
            # print (features[i])
    Features = features[idx_to_keep]
    Targets = targets[idx_to_keep]
    N0 = np.sum(Targets == 0)
    N1 = np.sum(Targets == 1)
    Nd = int (N0 - N1)
    idx0 = np.where(Targets == 0)[0]
    idx0_to_drop = random.sample( list(idx0), Nd )
    New_Features = np.delete(Features, idx0_to_drop, axis = 0)
    New_Targets = np.delete(Targets, idx0_to_drop, axis = 0)
    
    return New_Features, New_Targets

             
def plot_u_x(Infos, lane, segment, t):
    window = tuple([int(t/100) * 100, int(t/100) * 100 + 100]) 
    trjs = Infos[lane][segment][window][t]
    xs = trjs[:,0] - segment[0]
    vs = trjs[:,2] 
    plt.scatter(xs, vs)
    plt.xlim(0, 200)
    plt.ylim(0, 70)
    plt.show()
    
def plot_q_k(qvk):
    qs1 = []
    ks1 = []
    qs0 = []
    ks0 = []
    for lane in qvk:
        for zone in qvk[lane]:
            for window in qvk[lane][zone]:
                q, v, k, ssd = qvk[lane][zone][window]
                
                if k >= 60:
                    qs1.append(q)
                    ks1.append(k)
                else:
                    qs0.append(q)
                    ks0.append(k)
    plt.scatter(ks1, qs1, color = "red", s = 1, label = "congested")
    plt.scatter(ks0, qs0, color = "blue", s = 1, label = "uncongested")
    plt.xlabel("Density veh/km")
    plt.ylabel("Flow veh/hr")
    plt.xlim(0, 120)
    plt.ylim(0, 2500)
    plt.legend()
    plt.savefig("qk", dpi = 300)
    
    plt.show()
    
if __name__ == "__main__":
    
    paths = [r"vehicle-trajectory-data\0820am-0835am\trajectories-0820am-0835am.csv", 
             r"vehicle-trajectory-data\0750am-0805am\trajectories-0750am-0805am.csv", 
             r"vehicle-trajectory-data\0805am-0820am\trajectories-0805am-0820am.csv" 
              ]
    
    
   
    # 
    Features50k60c, Targets50k60c = [load_data("./data/Features50k60c3"), load_data("./data/Targets50k60c3")]
    Features50k60cnd, Targets50k60cnd = (Features50k60c[:,:3, :,:], Targets50k60c)
    NewFeatures50k60cnd, NewTargets50k60cnd = Clear_Features(Features50k60cnd, Targets50k60cnd) 
    # results50k60cnd = cross_validation(NewFeatures50k60cnd, NewTargets50k60cnd, rate = 0.8, just_one_trial = False)
    
    
    
    
    
    
    # Features50k60c, Targets50k60c = [load_data("./data/Features50k60c3"), load_data("./data/Targets50k60c3")]
    # Features50k60cndh1, Targets50k60cndh1 = (Features50k60c[:,:1, :,:], Targets50k60c)
    # NewFeatures50k60cndh1, NewTargets50k60cndh1 = Clear_Features(Features50k60cndh1, Targets50k60cndh1) 
    # results50k60cndh1 = cross_validation(NewFeatures50k60cndh1, NewTargets50k60cndh1, rate = 0.8, just_one_trial = False)
    
    
    
    
    # path = r"vehicle-trajectory-data\0820am-0835am\trajectories-0820am-0835am.csv"
    # us101_data = pd.read_csv(path)
    # Infos100, qvk100, wzveh100 = get_multiple_snapshots(us101_data, [1000,5000], segment = [0, 2400], window_size = 100, zone_size = 200, mpr = 100)

    # IInfos100q = get_multiple_infos(Infos100)
    # IIInfos100q = clean_multiple_infos(IInfos100q)
    # features100q, targets100q = get_features(IIInfos100q, qvk100, history = 3, var = 2, boundary = [60])
    # print (np.isnan(featuresq[:,:,:,1]).any())
    # print (np.max(featuresq[:,:,:,1]))
    
    # Infos50, qvk100, wzveh50 = get_multiple_snapshots(us101_data, [1000,5000], segment = [0, 2400], window_size = 100, zone_size = 200, mpr = 50)
    # IInfos50q = get_multiple_infos(Infos50)
    # IIInfos50q = clean_multiple_infos(IInfos50q)
    
    # plot_lever_factor_vs_density(IIInfos100q, qvk100, 1, tuple([0, 200]), [1000,5000], mpr = 100)
    # plot_lever_factor_vs_density(IIInfos50q, qvk100, 1, tuple([0, 200]), [1000,5000], mpr = 50)
    # plot_q_k(qvk100)
    # plot_u_x(Infos100, 4, tuple([600, 800]), 2770)
     
    
    
    # path = r"vehicle-trajectory-data\0820am-0835am\trajectories-0820am-0835am.csv"
    # us101_data = pd.read_csv(path)
    # Infos50, qvk50, wzveh50 = get_multiple_snapshots(us101_data, [1000,5000], segment = [0, 2400], window_size = 100, zone_size = 200, mpr = 50)
    # IInfos50q = get_multiple_infos(Infos50)
    # IIInfos50q = clean_multiple_infos(IInfos50q)
    # features50q, targets50q = get_features(IIInfos50q, qvk50, history = 3, var = 2, boundary = [60])
    
    # plot_lever_factor_vs_density(IIInfos50q, qvk50, 4, tuple([800, 1000]), [1000,5000], smooth = False)
    

    
    
    
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


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
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import f1_score
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
    speed = flow/density 
    ran = ran if not math.isnan(ran) else 0
    return flow, speed, density, ssd, ran




def get_multiple_snapshots_segments( data, timespan = [1500, 4500], segments = [[800,2400],[850,2450],[900,2500],[950,2550],[1000,2600]], window_size = 100, zone_size = 200, mpr = 100 ):
    
    Infos = {}
    wzveh = {}
    qvk = {}
    Infos2 = {}
    
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
            
def get_multiple_infos(Infos, window_size = 100, var = 3):
    
    IInfos = {}
    IInfos2 = {}
    IInfos3 = {}
    ks = []
    qs = []
    
    for lane in Infos:
        info = Infos[lane]
        if lane not in IInfos.keys():
            IInfos[lane] = {}
            IInfos2[lane] = {}
            IInfos3[lane] = {}
        for zone in info:
            zone_size = zone[1] - zone[0]
            subx = 7
            sub_lenx = 50
            strapx = (zone_size - sub_lenx)/(subx - 1)
            subzones = [ tuple([zone[0] + strapx * i, zone[0] + strapx * i + sub_lenx]) for i in range(subx) ]
            if zone not in IInfos[lane]:
                IInfos[lane][zone] = {}
                IInfos2[lane][zone] = {}
                IInfos3[lane][zone] = {}
            for window in info[zone]:
                window_size = window[1] - window[0]
                subt = 7
                sub_lent = 25
                strapt = (window_size - sub_lent)/(subt - 1)
                subwindows = [ tuple([window[0] + strapt * i, window[0] + strapt * i + sub_lent]) for i in range(subt) ]
                trjs = Infos[lane][zone][window]
                res = get_flow_speed_density(Infos[lane][zone][window], window, zone)
                ks.append(res[2])
                qs.append(res[0])
                U = res[1]
                
                dict_x = { }
                dict_t = { }
                for veh in trjs:
                    for i in range(len(trjs[veh]["t"])):
                        t = trjs[veh]["t"][i]
                        x = trjs[veh]["x"][i]
                        for sz in subzones:
                            for sw in subwindows:
                                if sz not in dict_x.keys():
                                    dict_x[sz] = { }
                                if sw not in dict_t.keys():
                                    dict_t[sw] = { }
                                if sz[0] <= x < sz[1]:
                                    if veh not in dict_x[sz].keys():
                                        dict_x[sz][veh] = {"x":[] , "t":[]}
                                    dict_x[sz][veh]["x"].append(x)
                                    dict_x[sz][veh]["t"].append(t)
                                
                                if sw[0] <= t < sw[1]:
                                    
                                    if veh not in dict_t[sw].keys():
                                        dict_t[sw][veh] = {"x":[] , "t":[]}
                                    dict_t[sw][veh]["x"].append(x)
                                    dict_t[sw][veh]["t"].append(t)
            
                vx = np.zeros(subx)
                vt = np.zeros(subt)
                for i in range(subx):
                    resx = get_flow_speed_density(dict_x[subzones[i]], window, subzones[i])
                    vx[i] = resx[1] if not math.isnan(resx[1]) else 0
                for i in range(subt):
                    rest = get_flow_speed_density(dict_t[subwindows[i]], subwindows[i], zone)  
                    vt[i] = rest[1] if not math.isnan(rest[1]) else 0
                indiceX = np.where(vx > 0)[0]
                velocityX = vx[indiceX]
                # print (indiceX, velocityX)
                if indiceX.shape[0] > 1:
                    regX = LinearRegression().fit(indiceX.reshape(-1,1), velocityX)
                    dU_dX = regX.coef_[0] 
                else:
                    dU_dX = 0
                
                indiceT = np.where(vt > 0)[0]
                velocityT = vt[indiceT]
                print (velocityT)
                if indiceT.shape[0] > 1:
                    regT = LinearRegression().fit(indiceT.reshape(-1,1), velocityT)
                    dU_dT = regT.coef_[0] 
                else:
                    dU_dT = 0
                    
                arr = np.zeros(3)
                arr[0] = min(U/60, 1)
                # arr[1] = dU_dX
                # arr[2] = dU_dT
                arr[1] = max(min(dU_dX/5, 1), -1)
                arr[2] = max(min(dU_dT/5, 1), -1)
                
                IInfos[lane][zone][window] = arr
                IInfos2[lane][zone][window] = dict_x
                IInfos2[lane][zone][window] = dict_t
                
    plt.scatter(ks, qs, s=0.1, color = "grey")
    return IInfos                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            
            
def clean_multiple_infos(IInfos, window_size = 100, var = 3):
    
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
    
    arr = np.zeros(var, dtype = "float")

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


def load_training_data_k(info, res, history = 3, var = 3, boundary = [45, 70], zone_size = 200, freeflows = True):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys())
    m = len(windows) - history
    features = np.zeros(shape = ((len(zones)) * m, history  * var))
    targets = np.zeros(shape = ((len(zones)) * m))
    c = 0

    for zone in info:
        for w in range(history, len(windows)):
            window = windows[w]
            if window not in list(res[zone].keys()):
                continue
                
            if freeflows == False and res[zone][window][2] < 30:
                continue
            
            cc = 0

            for v in range(var):
                for h in range(history):
                    features[c][cc] = info[zone][windows[w - h - 1]][v]
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

          
def get_features(IIInfos, ress, history = 3, var = 3, boundary = [45, 70], mode = "density", freeflows = True):

    Features = np.full((0, history * var), 0.0, dtype = "float")
    Targets = np.full(0, 0.0, dtype = "float")
    for lane in IIInfos:
        info = IIInfos[lane]
        res = ress[lane]
        features, targets = load_training_data_k(info, res, history, var, boundary, freeflows = freeflows) 
        Features = np.append(Features, features, axis = 0)
        Targets = np.append(Targets, targets, axis = 0)
        # print (np.where(Targets == 1))
    return Features, Targets

            
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
        
        self.fc1 = nn.Linear(9, 4)
        self.fc2 = nn.Linear(4, 2)
        # self.fc2 = nn.Linear(4, 2)
        # self.fc2 = nn.Linear(3, 2)
        
    def forward(self, input):
        
        # print (input.shape)
        output = torch.tensor(input, dtype=torch.float)
        # print (output.shape)
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
    test_cms = []
    test_f1 = []
    for i in range( int(1/(1-rate)) ): 
        trainset = MyDataset(features[train_ids[i]], targets[train_ids[i]])
        validset = MyDataset(features[valid_ids[i]], targets[valid_ids[i]])
        testset = MyDataset(features[test_ids[i]], targets[test_ids[i]])
        if mode == "offline":
            _model,_loss,_accuracy,_test_accuracy, _test_cm, _test_f1 = run_NN(NN(),running_mode='train', train_set=trainset, valid_set=validset, test_set=testset, 
                                                           batch_size=1, learning_rate=0.01, n_epochs=n_epochs, stop_thr=stop_thr, shuffle=True)
        elif mode == "online":
            _model,_loss,_accuracy,_test_accuracy, _test_cm, _test_f1 = run_NN_online(NN(),running_mode='train', train_set=trainset, valid_set=validset, test_set=testset, 
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
    return accuracies, _model, all_accuracy, test_accuracies, test_cms, round(np.mean(test_f1),2 )

def supplement_features(paths, Features, Targets, timespan = [900, 4900], segment = [900, 2300], keep = [1, 2], times = 1, mpr = 100, boundary = [40, 60], mode = "density", history = 3, window_size = 100, zone_size = 200):
    for t in range(times):
        path = paths[t]
        data = pd.read_csv(path)
        Infos, qvk, wzveh = get_multiple_snapshots(data, timespan, segment, window_size, zone_size, mpr)
        IInfos = get_multiple_infos(Infos)
        IIInfos = clean_multiple_infos(IInfos)
        features, targets = get_features(IIInfos, qvk, history, var=3, boundary=boundary)
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


def _smooth_(y, box_pts=100):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_lever_factor_vs_density(Info, QVK, lane, zone, timespan, smooth = False):
    
    fs = []
    ks = []
    trjs = Info[lane][zone]
    stats = QVK[lane][zone]
    fig, ax = plt.subplots()
    windows = [tuple([w, w+100]) for w in np.arange(timespan[0], timespan[1], 100)]
    steps = []
    for window in windows:
        ks.append(stats[window][2])
        fs.append(trjs[window][3])
        steps.append(window[0] + 50)
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
    ax2.scatter(steps, fs, color = "red", s = 20)
    ax2.plot(steps, fs,  color = "red", linestyle = "dashed", linewidth = 1)
    ax2.plot(steps, np.zeros(len(steps)), linestyle ="dashed", color = "grey")
    ax2.set_ylabel("Slope", color = "red")
    ax2.set_ylim(-1.3, 1.3)
    # plt.xlim(1000, 5000)
    plt.title("u-x slope changes vs actual density changes")
    plt.show()
    
    # fs = []
    # ks = []
    # qs = []
    # trjs = Info[lane][zone]
    # stats = QVK[lane][zone]
    # windows = [tuple([w, w+100]) for w in np.arange(timespan[0], timespan[1], 100)]
    # for window in windows:
    #     arr = trjs[window]
    #     for t in range(arr.shape[0]):
    #         fs.append(arr[t][3])
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
    # plt.title("Flow-Density Relationship")
    # plt.show()
    # fs = []
    # ks = []
    # us = []
    # windows = list(trjs.keys())
    # steps = [ ss[0]+50 for ss in windows ]
    # frames = np.arange(windows[0][0], windows[-1][1])
    # for window in trjs:
    #     arr = trjs[window]
    #     for t in range(arr.shape[0]):
    #         fs.append(arr[t][3])
    #     ks.append(stats[window][2])   
    #     us.append(stats[window][1])   
    # fs = np.asarray(fs)
    # us = np.asarray(us)
    # fig, ax = plt.subplots()
    # ax.scatter(steps, ks, color = "blue")
    # ax.plot(steps, ks, color = "black", linestyle = "dashed")
    # ax.set_xlabel("Frames (0.1sec)")
    # ax.set_ylabel("Density (veh/km)", color = "blue")
    # ax.set_ylim(0, 100)
    # ax.set_xlim(1000, 5000)
    
    # ax2 = ax.twinx()
    # if smooth:
    #     fs = _smooth_(fs)
    # # print (fs2 == fs.all())
    # ax2.plot(frames, fs, linewidth = 0.2, color = "red")
    # ax2.plot(frames, np.zeros(len(frames)), linestyle ="dashed", color = "grey")
    # ax2.set_ylabel("Slope", color = "red")
    # ax2.set_ylim(-0.5, 0.5)
    # # plt.xlim(1000, 5000)
    # plt.title("u-x slope changes vs actual density changes")
    # plt.show()
    
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
        features, targets = get_features(IIInfos, qvk, history, var=3, boundary=boundary, mode=mode, freeflows=freeflows)
        # print (features.shape)
        for ke in keep:
            idx = np.where(targets == ke)[0]
            features1 = features[idx]
            targets1 = targets[idx]
            Features = np.append(Features, features1, axis = 0)
            Targets = np.append(Targets, targets1, axis = 0) 
    return Features, Targets


def Generate_Features(Paths, timespan, segments, mpr, boundary, window_size=100, zone_size=200, mode="density", var=3, history = 3, one_iter = False, freeflows = True):
    Features = np.full((0, history * var), 0.0, dtype = "float")
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
    f1 = []
    for i in range( int(1/(1-rate)) ): 
        train_Features = features[train_ids[i]]
        train_Targets = targets[train_ids[i]]
        test_Features = np.append(features[valid_ids[i]], features[test_ids[i]], axis = 0)
        test_Targets = np.append(targets[valid_ids[i]], targets[test_ids[i]], axis = 0)
        rf = RandomForestClassifier(max_depth=10, random_state=0)
        rf.fit(train_Features, train_Targets)
        rfpreds = rf.predict(test_Features)
        rfcm = confusion_matrix(rfpreds, test_Targets)
        rfacc = round ((rfcm[0][0] + rfcm[1][1]) / (rfcm[0][0] + rfcm[1][1] + rfcm[0][1] + rfcm[1][0]) * 100, 2)
        test_cms.append(rfcm)
        test_accuracies.append(rfacc)
        f1.append(f1_score(test_Targets, rfpreds))
    return test_cms, round(np.mean(f1) * 100, 2)


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
    f1 = []
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
        f1.append(f1_score(test_Targets, svmpreds))
    return test_cms, round(np.mean(f1) * 100, 2)

def Cross_Validation_Logistics(features, targets, rate = 0.8):
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
        logit = LogisticRegression()
        logit.fit(train_Features, train_Targets)
        logitpreds = logit.predict(test_Features)
        logitcm = confusion_matrix(logitpreds, test_Targets)
        logitacc = round ((logitcm[0][0] + logitcm[1][1]) / (logitcm[0][0] + logitcm[1][1] + logitcm[0][1] + logitcm[1][0]) * 100, 2)
        test_cms.append(logitcm)
        test_accuracies.append(logitacc)
    return test_cms, np.mean(np.asarray(test_accuracies))

    
def Extract_nonZero_Features(features, targets, var = 3, history = 1):
    idx_to_keep = []
    for f in range(features.shape[0]):
        if np.all(features[f] == 0)  or np.isnan(features[f]).any():
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
    Ks0 = []
    Qs0 = []
    Ks1 = []
    Qs1 = []
    for lane in IIInfos:
        for zone in IIInfos[lane]:
            for window in IIInfos[lane][zone]:
                try:
                    res = qvk[lane][zone][window]
                    if res[2] >= 60:
                        Ks1.append(res[2])
                        Qs1.append(res[0])
                    else:
                        Ks0.append(res[2])
                        Qs0.append(res[0])
                except KeyError:
                    pass
                
    plt.scatter(Ks0, Qs0, s=0.1, color="blue")
    plt.scatter(Ks1, Qs1, s=0.1, color="red")
    plt.xlim(0, 140)
    plt.ylim(0, 2600)
    plt.xlabel("Density (veh/km)")
    plt.ylabel("Flow (veh/h)")
    plt.show()
    
def get_coefficients_vs_density(IIInfos, ress, history = 3, var = 3, boundary = [45, 70], mode = "density", freeflows = True):

    Features = np.full((0, history * var), 0.0, dtype = "float")
    Targets = np.full(0, 0.0, dtype = "float")
    for lane in IIInfos:
        info = IIInfos[lane]
        res = ress[lane]
        features, targets = load_visualize_data_k(info, res, history, var, boundary, freeflows = freeflows) 
        Features = np.append(Features, features, axis = 0)
        Targets = np.append(Targets, targets, axis = 0)
        # print (np.where(Targets == 1))
    # plt.scatter(Features[:,2], Targets, s=0.1, color = "red")    
    # plt.hist(Features[:,1][np.where(Targets == -1)], color = "green" , edgecolor = 'black', bins = 60, density = True )
    # plt.hist(Features[:,1][np.where(Targets == 1)], color = "orange" , edgecolor = 'black', bins = 60, density = True )
   
    # plt.hist(Features[:,2][np.where(Targets == 0)], color = "crimson" , edgecolor = 'black', bins = 80, density = True )
    # plt.hist(Features[:,2][np.where(Targets == -1)], color = "green" , edgecolor = 'black', bins = 60, density = True )
    # plt.hist(Features[:,2][np.where(Targets == 1)], color = "orange" , edgecolor = 'black', bins = 60, density = True )
    
    return Features, Targets  
    
def load_visualize_data_k(info, res, history = 3, var = 3, boundary = [45, 70], zone_size = 200, freeflows = True):
    zones = list(info.keys())
    windows = list(info[zones[0]].keys())
    m = len(windows) - history
    features = np.zeros(shape = ((len(zones)) * m, history  * var))
    targets = np.zeros(shape = ((len(zones)) * m))
    c = 0

    for zone in info:
        for w in range(history, len(windows)):
            window = windows[w]
            if window not in list(res[zone].keys()):
                continue
                
            if freeflows == False and res[zone][window][2] < 30:
                continue
            
            cc = 0

            for v in range(var):
                for h in range(history):
                    features[c][cc] = info[zone][windows[w - h - 1]][v]
                    cc += 1
   
            try:
                k1 = res[zone][window][2] 
                k0 = res[zone][windows[w - 1]][2] 
            except KeyError:
                k1 = 0
                k0 = 0
            if k1 == 0 or k0 == 0:
                continue
            
            # targets[c] = k1 - k0
            if k1 >= 60 and k0 < 60:
                targets[c] = 1
            elif k1 < 60 and k0 >= 60:
                targets[c] = -1
            else:
                targets[c] = 0
                
            # print (features[c])
            # print (targets[c])
            c += 1
            
    return features, targets
    
def Z_Test(sample1, sample2):
    n1 = sample1.shape[0]
    n2 = sample2.shape[0]
    v1 = np.var(sample1)
    v2 = np.var(sample2)
    m1 = np.mean(sample1)
    m2 = np.mean(sample2)
    
    print (n1, n2, v1, v2, m1, m2)
    return abs(m2 - m1) / math.sqrt( v1/n1 + v2/n2  )

def Plot_Contour(Features, Targets):
    intervals = np.arange(-1.00, 1.00, 0.1)
    distributions = np.zeros(shape = (intervals.shape[0], intervals.shape[0]))
    for feature in Features:
        sp = int((feature[1] + 1)/0.1) if int((feature[1] + 1)/0.1) < 20 else 19
        a = int((feature[2] + 1)/0.1) if int((feature[2] + 1)/0.1) < 20 else 19
        distributions[sp][a] += 1
    distributions /= Targets.shape[0]
    
    # from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection="3d")
    X, Y = np.mgrid[-1:1:20j, -1:1:20j]
    Z = distributions
    ax.contour(X,Y, Z, colors='black')
    plt.show()
    # ax.plot_surface(X, Y, Z, cmap="winter_r", linewidth = 5, rstride=1, cstride=1, antialiased=True, alpha = 0.8)
    # ax.plot_wireframe(X, Y, Z, color ='black', linewidth = 0.5, alpha = 1)
    # ax.set_xticks(np.arange(-1, 1.5, 0.5))
    # ax.set_yticks(np.arange(-1, 1.5, 0.5))
    # plt.savefig('3dcontour.png', dpi=300)
    # plt.show()
    
    return distributions
    
    
    
if __name__ == "__main__":
    
    paths = [r"vehicle-trajectory-data\0820am-0835am\trajectories-0820am-0835am.csv", 
             r"vehicle-trajectory-data\0750am-0805am\trajectories-0750am-0805am.csv", 
             r"vehicle-trajectory-data\0805am-0820am\trajectories-0805am-0820am.csv" 
              ]

    segments = [[800,2000],[850,1850],[900,1900],[950,1950],[825,1825],[875,1875],[925,1925],[975,1975]]
    
    
   
    ''' 
    Run the following scripts to get the result in 100% connectivity
    '''
    # Features100k60stnd, Targets100k60stnd, IIInfos100k60stnd = Generate_Features(paths,[1000,5000],segments,100,[60],one_iter = False,history=1,zone_size=200)
    
    # NewFeatures100k60stnd, NewTargets100k60stnd = Extract_nonZero_Features(Features100k60st,Targets100k60st)
    # NewFeatures100k60stnd, NewTargets100k60stnd = Drop_Features(NewFeatures100k60stnd, NewTargets100k60stnd)
    # outcomesSVM100k60STnd = Cross_Validation_SVM(NewFeatures100k60stnd, NewTargets100k60stnd, rate = 0.8)
    # outcomesRF100k60STnd = Cross_Validation_Random_Forest(NewFeatures100k60stnd, NewTargets100k60stnd, rate = 0.8)
    
    # save_data(Features100k60st, "Features100k60stnd")
    # save_data(Targets100k60st, "Targets100k60stnd")
    
    
    # Features100k60stnd_3, Targets100k60stnd_3, IIInfos100k60stnd_3 = Generate_Features(paths,[1000,5000],segments,100,[60],one_iter = False,history=3,zone_size=200)
    # NewFeatures100k60stnd_3, NewTargets100k60stnd_3 = Extract_nonZero_Features(Features100k60stnd_3,Targets100k60stnd_3)
    # NewFeatures100k60stnd_3, NewTargets100k60stnd_3 = Drop_Features(NewFeatures100k60stnd_3, NewTargets100k60stnd_3)
    outcomesSVM100k60STnd_3 = Cross_Validation_SVM(NewFeatures100k60stnd_3, NewTargets100k60stnd_3, rate = 0.8)
    outcomesRF100k60STnd_3 = Cross_Validation_Random_Forest(NewFeatures100k60stnd_3, NewTargets100k60stnd_3, rate = 0.8)
    
    
    # save_data(Features100k60stnd_3, "Features100k60stnd_3")
    # save_data(Targets100k60stnd_3, "Targets100k60stnd_3")
    
    # Features100k60st = load_data("Features100k60stnd")
    # Targets100k60st = load_data("Targets100k60stnd")
    
    # save_data(Features100_k60z, "Features100k60z")
    # save_data(Targets100_k60z, "Targets100k60z")
    '''
    Run the following scripts to get the result in 50% connectivity
    '''
    # Features50k60st, Targets50k60st, IIInfos50k60st = Generate_Features(paths,[1000,5000],segments,50,[60],one_iter = False,history=1,zone_size=200, freeflows = True)
    
    # NewFeatures50k60st, NewTargets50k60st = Extract_nonZero_Features(Features50k60st, Targets50k60st)
    # NewFeatures50k60st, NewTargets50k60st = Drop_Features(NewFeatures50k60st, NewTargets50k60st)
    
    # save_data(Features50k60st, "Features50k60stnd")
    # save_data(Targets50k60st, "Targets50k60stnd")
    
    # Features50k60stnd_3, Targets50k60stnd_3, IIInfos50k60stnd_3 = Generate_Features(paths,[1000,5000],segments,50,[60],one_iter = False,history=3,zone_size=200)
    
    # NewFeatures50k60stnd_3, NewTargets50k60stnd_3 = Extract_nonZero_Features(Features50k60stnd_3,Targets50k60stnd_3)
    # NewFeatures50k60stnd_3, NewTargets50k60stnd_3 = Drop_Features(NewFeatures50k60stnd_3, NewTargets50k60stnd_3)
    # outcomesSVM50k60STnd_3 = Cross_Validation_SVM(NewFeatures50k60stnd_3, NewTargets50k60stnd_3, rate = 0.8)
    # outcomesRF50k60STnd_3 = Cross_Validation_Random_Forest(NewFeatures50k60stnd_3, NewTargets50k60stnd_3, rate = 0.8)
    # outcomesNN50k60STnd_3 = cross_validation(NewFeatures50k60stnd_3, NewTargets50k60stnd_3, rate = 0.8, just_one_trial = False, n_epochs=3*10**4, stop_thr=1e-8, mode = "offline")
    
    # save_data(Features50k60stnd_3, "Features50k60stnd_3")
    # save_data(Targets50k60stnd_3, "Targets50k60stnd_3")
    
    # Features50k60st, Targets50k60st = [load_data("Features50k60stnd"), load_data("Targets50k60stnd")]
    # Features50k60z, Targets50k60z = [ load_data("Features50k60z"), load_data("Targets50k60z")  ]
    
    # outcomesLogit50k60STnd = Cross_Validation_Logistics(NewFeatures50k60st, NewTargets50k60st, rate = 0.8)
    # outcomesSVM50k60STnd = Cross_Validation_SVM(NewFeatures50k60st, NewTargets50k60st, rate = 0.8)
    # outcomesRF50k60STnd = Cross_Validation_Random_Forest(NewFeatures50k60st, NewTargets50k60st, rate = 0.8)
    # outcomesNN50k60STnd = cross_validation(NewFeatures50k60st, NewTargets50k60st, rate = 0.8, just_one_trial = False, n_epochs=3*10**4, stop_thr=1e-8, mode = "offline")
    
    # Features50k60z, Targets50k60z = [ load_data("Features50k60z"), load_data("Targets50k60z")  ]
    # NewFeatures50k60z, NewTargets50k60z = Extract_nonZero_Features(Features50k60z,Targets50k60z)
    # NewFeatures50k60z, NewTargets50k60z = Drop_Features(NewFeatures50k60z, NewTargets50k60z)
    # outcomesNN50k60Z = cross_validation(NewFeatures50k60z, NewTargets50k60z, rate = 0.8, just_one_trial = False, n_epochs=3*10**5, stop_thr=1e-6, mode = "offline")
    
    # trainset = MyDataset(NewFeatures50k60z, NewTargets50k60z)
    # validset = MyDataset(NewFeatures50k60z, NewTargets50k60z)
    # testset = MyDataset(NewFeatures50k60z, NewTargets50k60z)
    # results50k60z = run_NN(model50k60z,running_mode='test', train_set=trainset, valid_set=validset, test_set=testset, 
    #                                                         batch_size=1, learning_rate=0.01, n_epochs=1*10**5, stop_thr=1e-12, shuffle=True)
    
    '''
    Run the following scripts to get the result in 30% connectivity
    '''
    
    # Features30k60st, Targets30k60st, IIInfos30k60st = Generate_Features(paths,[1000,5000],segments,30,[60],one_iter = False,history=1,zone_size=200, freeflows = True)
    
    # NewFeatures30k60st, NewTargets30k60st = Extract_nonZero_Features(Features30k60st, Targets30k60st)
    # NewFeatures30k60st, NewTargets30k60st = Drop_Features(NewFeatures30k60st, NewTargets30k60st)
    # outcomesLogit30k60STnd = Cross_Validation_Logistics(NewFeatures30k60st, NewTargets30k60st, rate = 0.8)
    # outcomesSVM30k60STnd = Cross_Validation_SVM(NewFeatures30k60st, NewTargets30k60st, rate = 0.8)
    # outcomesRF30k60STnd = Cross_Validation_Random_Forest(NewFeatures30k60st, NewTargets30k60st, rate = 0.8)
    # outcomesNN30k60STnd = cross_validation(NewFeatures30k60st, NewTargets50k60st, rate = 0.8, just_one_trial = False, n_epochs=3*10**5, stop_thr=1e-6, mode = "offline")
    
    
    # save_data(Features30k60st, "Features30k60stnd")
    # save_data(Targets30k60st, "Targets30k60stnd")
    
    # Features30k60stnd_3, Targets30k60stnd_3, IIInfos30k60stnd_3 = Generate_Features(paths,[1000,5000],segments,30,[60],one_iter = False,history=3,zone_size=200)
    
    # NewFeatures30k60stnd_3, NewTargets30k60stnd_3 = Extract_nonZero_Features(Features30k60stnd_3,Targets30k60stnd_3)
    # NewFeatures30k60stnd_3, NewTargets30k60stnd_3 = Drop_Features(NewFeatures30k60stnd_3, NewTargets30k60stnd_3)
    # outcomesSVM30k60STnd_3 = Cross_Validation_SVM(NewFeatures30k60stnd_3, NewTargets30k60stnd_3, rate = 0.8)
    # outcomesRF30k60STnd_3 = Cross_Validation_Random_Forest(NewFeatures30k60stnd_3, NewTargets30k60stnd_3, rate = 0.8)
    # outcomesNN30k60STnd_3 = cross_validation(NewFeatures30k60stnd_3, NewTargets30k60stnd_3, rate = 0.8, just_one_trial = False, n_epochs=3*10**4, stop_thr=1e-8, mode = "offline")
    
    
    # save_data(Features30k60stnd_3, "Features30k60stnd_3")
    # save_data(Targets30k60stnd_3, "Targets30k60stnd_3")
    
    
    
    
    
    
    
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
    
    
    # outcomesRF50k60Z2 = Cross_Validation_Random_Forest(NewFeatures50k60z2, NewTargets50k60z2, rate = 0.8)
    # acc50k60z2, model50k60z2, accuracy50k60z2, test_accuracy50k60z2, cms50_k60z2 = cross_validation(NewFeatures50k60z2, NewTargets50k60z2, rate = 0.8, just_one_trial = False, n_epochs=3*10**5, stop_thr=1e-6, mode = "offline")
    
    '''
    See the intermediate variables of the model
    '''
    # us101_data = pd.read_csv(paths[0])
    # Infos100, qvk100, wzveh100 = get_multiple_snapshots_segments(us101_data, [1000,5000], segments = segments, window_size = 100, zone_size = 200, mpr = 100)
    # IInfos100 = get_multiple_infos(Infos100)
    # IIInfos100 = clean_multiple_infos(IInfos100)
    # featuresv, targetsv = get_coefficients_vs_density(IIInfos100, qvk100, history = 1, var = 3, boundary = [60])
    # 
    
    
    # features, targets = get_features(IIInfos100, qvk100, history = 1, var = 4, boundary = [60])
    # features, targets =  Extract_nonZero_Features(features, targets)
    # plot_lever_factor_vs_density(IIInfos100, qvk100, 4, tuple([1400, 1600]), [1000, 5000], smooth = False)
    
    # us101_data = pd.read_csv(paths[0])
    # Infos50, qvk50, wzveh50 = get_multiple_snapshots_segments(us101_data, [1000,5000], segments = segments, window_size = 100, zone_size = 200, mpr = 50)
    # IInfos50 = get_multiple_infos(Infos50)
    # IIInfos50 = clean_multiple_infos(IInfos50)
    # features, targets = get_features(IIInfos50, qvk50, history = 1, var = 3, boundary = [60]) 
    # features, targets =  Extract_nonZero_Features(features, targets)
    # plt.hist(features[:,1], color = "orange" , edgecolor = 'black', bins = 80, density = True)
    # plt.xlim(-6,6)
    # plt.ylim(0, 0.55)
    # plt.xlabel("Propagation Coefficient")
    # plt.ylabel("Probability Density")
    # plt.show()
    # plt.hist(features[:,2], color = "green" , edgecolor = 'black', bins = 80, density = True)
    # plt.xlim(-6,6)
    # plt.ylim(0, 0.55)
    # plt.xlabel("Acceleration Coefficient")
    # plt.ylabel("Probability Density")
    # plt.show()
    # print (np.std(features[:,1]), np.std(features[:,2]))
    # plt.scatter(features[:,1],features[:,2],s=0.1, color="black" )
    # plt.show()
    # plt.hist(Features50k60st[:,1], color = "grey" , bins = 80, density = True)
    # plt.plot()
    # plt.hist(Features50k60st[:,2], color = "grey" , bins = 80, density = True)
    # plt.plot()
    
    # Plot_States(IIInfos50, qvk50)
    
    # plot_lever_factor_vs_density(IIInfos50, qvk50, 4, tuple([1400, 1600]), [1000, 5000], smooth = False)
    # print (np.max(features[:,1]), np.max(features[:,2]))
    
    # plt.hist(NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 0)], color = "limegreen" , edgecolor = 'black', bins = 60, density = True, label = "Uncongested in next 10s", alpha = 0.5 )
    # plt.hist(NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 1)], color = "crimson" , edgecolor = 'black', bins = 40, density = True, label = "Congested in next 10s", alpha = 0.5 )
    # plt.ylabel("Probability Density")
    # plt.xlabel("Propagation Coefficient")
    # plt.legend()
    # plt.ylim(0, 4)
    # plt.xlim(-0.9, 0.9)
    # plt.savefig('rg.sp.png', dpi=300)
    # plt.show()
    
    
    
    # z_sp = Z_Test(NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 1)], NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 0)])
    # z_a = Z_Test(NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 1)], NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 0)])
    # z_sp = ( np.mean(NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 1)]) - np.mean(NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 0)]) ) / \
    #     (np.var(NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 1)]) + (np.var(NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 0)])))
    # z_a = ( np.mean(NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 1)]) - np.mean(NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 0)]) ) / \
    #     (np.var(NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 1)]) + (np.var(NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 0)])))
    
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # import numpy as np
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.scatter(
    #     NewFeatures100k60stnd[:,0][np.where(NewTargets100k60stnd == 0)],
    #     NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 0)],
    #     NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 0)],
    #     color = "blue",
    #     marker = "o",
    #     label = "uncongested in next 10s",
    #     s = 1
    #     )
    
    # ax.scatter(
    #     NewFeatures100k60stnd[:,0][np.where(NewTargets100k60stnd == 1)],
    #     NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 1)],
    #     NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 1)],
    #     color = "red",
    #     marker = "^",
    #     label = "congested in next 10s",
    #     s = 1
    #     )
    
    # ax.set_yticks(np.arange(-1, 1.5, 0.5), minor=False)
    # ax.set_zticks(np.arange(-1, 1.5, 0.5), minor=False)
    # ax.set_xlabel("Normalized Average Speed", fontsize = 5)
    # ax.set_ylabel("Normalized Propagation Coefficent", fontsize = 5)
    # ax.set_zlabel("Normalized Acceleration Coefficient", fontsize = 5)
    # plt.legend()
    # # plt.savefig('3d.png', dpi=300)
    # plt.show() 
    
    # plt.scatter(
    #     NewFeatures100k60stnd[:,0][np.where(NewTargets100k60stnd == 0)],
    #     NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 0)],
    #     # NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 0)],
    #     color = "blue",
    #     marker = "o",
    #     label = "uncongested in next 10s",
    #     s = 1
    #     )
    # plt.scatter(
    #     NewFeatures100k60stnd[:,0][np.where(NewTargets100k60stnd == 1)],
    #     NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 1)],
    #     # NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 1)],
    #     color = "red",
    #     marker = "^",
    #     label = "congested in next 10s",
    #     s = 1
    #     )
    # plt.show()
    
    '''
    one way ANOVA
    '''
    # from scipy.stats import f_oneway
    # p_coef_1 = NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 1)][500:1300]
    # a_coef_1 = NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 1)][500:1300]
    # p_coef_0 = NewFeatures100k60stnd[:,1][np.where(NewTargets100k60stnd == 0)][500:1300]
    # a_coef_0 = NewFeatures100k60stnd[:,2][np.where(NewTargets100k60stnd == 0)][500:1300]
    
    # F_value_p, p_value_p = f_oneway(p_coef_1, p_coef_0)
    # F_value_a, p_value_a = f_oneway(a_coef_1, a_coef_0)
    # F_value, p_value = f_oneway( NewFeatures100k60stnd[np.where(NewTargets100k60stnd == 1)][500:1300]
    #                             ,NewFeatures100k60stnd[np.where(NewTargets100k60stnd == 0)][500:1300]   )
    
    
    # d = Plot_Contour(NewFeatures100k60stnd[np.where(NewTargets100k60stnd == 0)], NewTargets100k60stnd[np.where(NewTargets100k60stnd == 0)])
    # d = Plot_Contour(NewFeatures100k60stnd[np.where(NewTargets100k60stnd == 1)], NewTargets100k60stnd[np.where(NewTargets100k60stnd == 1)])
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


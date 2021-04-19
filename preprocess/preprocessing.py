# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import networkx as nx
import math
import sys
import matplotlib.pyplot as plt
import collections

class dataset():
    def __init__(self,data,vehicles):
        self.data = data
        self.vehicles = list(set(list(data["Vehicle_ID"])))
        
    def _key_(self,vehicle_index):
        count_index = 0
        for vehicle_key in self.vehicles:
            if count_index == vehicle_index:
                break
            count_index += 1
        return vehicle_key

    def _index_(self,vehicle_ID):
        count_index = 0
        for vehicle_key in self.vehicles:
            if vehicle_key == vehicle_ID:
                break
            count_index += 1
        return count_index
    
    def trajectory(self,i):
        D = {}
        D["Time"] = self.data[i:i+1]["Frame_ID"][i]
        D["Location"] = self.data[i:i+1]["Local_Y"][i]
        D["Speed"] = self.data[i:i+1]["v_Vel"][i]
        D["Lane"] = self.data[i:i+1]["Lane_ID"][i] 
        return D
    
    def data_by_vehicle(self):
        ID = 0
        V = {}
        DD = {}
        for i in range(len(self.data)-1):
            if ID != self.data[i:i+1]["Vehicle_ID"][i]:
                if len(DD) != 0:
                    V[ID] = DD
                    print ("Importing vehicle #"+str(ID)+" ...")
                ID = self.data[i:i+1]["Vehicle_ID"][i]
                DD = {}
                D = {}
                D["Time"] = self.data[i:i+1]["Frame_ID"][i]
                D["Location"] = self.data[i:i+1]["Local_Y"][i]
                D["Speed"] = self.data[i:i+1]["v_Vel"][i]
                D["Lane"] = self.data[i:i+1]["Lane_ID"][i]
                D["Acc"] = self.data[i:i+1]["v_Acc"][i]
                DD[self.data[i:i+1]["Frame_ID"][i]] = D
            else:
                D = {}
                D["Time"] = self.data[i:i+1]["Frame_ID"][i]
                D["Location"] = self.data[i:i+1]["Local_Y"][i]
                D["Speed"] = self.data[i:i+1]["v_Vel"][i]
                D["Lane"] = self.data[i:i+1]["Lane_ID"][i]
                D["Acc"] = self.data[i:i+1]["v_Acc"][i]
                DD[self.data[i:i+1]["Frame_ID"][i]] = D
        ID = self.data[i:i+1]["Vehicle_ID"][i]
        V[ID] = DD
        print ("Importing vehicle #"+str(ID)+" ...")
        print ("Hashtable Creation  Complete")
        return V
    
    def data_by_lane(self):
        V = {}
        V = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
        prev_Lane = self.data[0:1]["Lane_ID"][0]
        prev_ID = self.data[0:1]["Vehicle_ID"][0]
        DD = [self.trajectory(0)]
        for i in range(1, len(self.data)-1):
            Lane = self.data[i:i+1]["Lane_ID"][i]
            ID = self.data[i:i+1]["Vehicle_ID"][i]
            if (Lane == prev_Lane and ID == prev_ID):
                DD.append(self.trajectory(i))
            else:
                V[Lane].append(DD)
                print("Lane #"+str(Lane)+" Complete")
                DD = []
                DD.append(self.trajectory(i))
            print("Trajectory #"+str(i+1)+" Complete")
            prev_Lane = Lane
            prev_ID = ID
        V[Lane].append(DD)
        return V
        
    def read_vehicle(self,vehicle):
        s = pd.read_csv(r"vehicles\\"+str(vehicle)+".csv")
        s = s.values.T
        D = {}
        for i in range(len(s)-1):
            D[s[i+1][4]] = {"Location":round(s[i+1][2]/3.28084,3),"Speed":round(s[i+1][3]/3.28084,2),"Lane":s[i+1][1], "Acc":s[i+1][0]}
        return D


    def single_vehicle_timespace(self,trj,timespan):
        time = []
        space = []
        for trjj in trj:
            if (trjj > timespan[0] and trjj < timespan[1]):
                time.append(trjj)
                space.append(trj[trjj]["Location"])
        plt.plot(time,space)
    
    def multiple_vehicle_timespace(self,list_of_vehs,filename,timespan = [500,800]):
        for veh in list_of_vehs:
            trj = self.read_vehicle(veh)
            self.single_vehicle_timespace(trj,timespan)
        plt.show()
        plt.savefig(r"US_101_wave\ "+ filename+".png")
    
    def lane_timespace(self,lane):
        for trip in lane:
            time = []
            space = []
            for trj in trip:
                if (trj["Time"] > 1000 and trj["Time"] < 2000):
                    time.append(trj["Time"])
                    space.append(trj["Location"])
            plt.plot(time,space)
        saved = plt.figure()
        saved.savefig(r"US_101_wave\lane3_trajectory.png",dpi=1200)
    
    
    def get_preceding_convoy(self,lead_veh_key,commu_dist,timespan,convoy_size=15,by_lane=True):
        convoy = {}
        if by_lane:
            lead_index = self._index_(lead_veh_key)
            for t in range(timespan[0],timespan[1]+1):
                count = 1  
                try:
                    lead_veh = self.read_vehicle(lead_veh_key)[t]
                except FileNotFoundError:
                    print("Vehicle #" + str(lead_veh_key) + " not recorded!")
                    break
                except KeyError:
                    print("Timespan not covered!")
                    break
                temp = {}
                temp[lead_veh_key] = lead_veh
                for i in range(1,51):
                    if count >= convoy_size:
                        break;
                    try:
                        sear_veh_key = self._key_(lead_index + i)
                        sear_veh = self.read_vehicle(sear_veh_key)[t]
                        if 0 < lead_veh["Location"] - sear_veh["Location"] < commu_dist and sear_veh["Lane"] == lead_veh["Lane"]:
                            temp[sear_veh_key] = sear_veh
                            count += 1
                    except KeyError:
                        continue;
                if count < convoy_size:
                    for i in range(1,51):
                        if count >= convoy_size:
                            break;
                        try:
                            sear_veh_key = self._key_(lead_index - i)
                            sear_veh = self.read_vehicle(sear_veh_key)[t]
                            if 0 < lead_veh["Location"] - sear_veh["Location"] < commu_dist and sear_veh["Lane"] == lead_veh["Lane"]:
                                temp[sear_veh_key] = sear_veh
                                count += 1
                        except KeyError:
                            continue;
                convoy[t] = temp
                print("Time"+str(t)+": "+str(count)+" vehicles found in convoy")
            print("Complete")
        else:
            lead_index = self._index_(lead_veh_key)
            for t in range(timespan[0],timespan[1]+1):
                count = 1  
                try:
                    lead_veh = self.read_vehicle(lead_veh_key)[t]
                except FileNotFoundError:
                    print("Vehicle #" + str(lead_veh_key) + " not recorded!")
                    break
                except KeyError:
                    print("Timespan not covered!")
                    break
                temp = {}
                temp[lead_veh_key] = lead_veh
                for i in range(1,51):
                    if count >= convoy_size:
                        break;
                    try:
                        sear_veh_key = self._key_(lead_index + i)
                        sear_veh = self.read_vehicle(sear_veh_key)[t]
                        if 0 < lead_veh["Location"] - sear_veh["Location"] < commu_dist:
                            temp[sear_veh_key] = sear_veh
                            count += 1
                    except KeyError:
                        continue;
                if count < convoy_size:
                    for i in range(1,51):
                        if count >= convoy_size:
                            break;
                        try:
                            sear_veh_key = self._key_(lead_index - i)
                            sear_veh = self.read_vehicle(sear_veh_key)[t]
                            if 0 < lead_veh["Location"] - sear_veh["Location"] < commu_dist:
                                temp[sear_veh_key] = sear_veh
                                count += 1
                        except KeyError:
                            continue;
                convoy[t] = temp
                print("Time"+str(t)+": "+str(count)+" vehicles found in convoy")
            print("Complete")
        return convoy

def get_trajectory_by_lane_loc_veh(trajectories):
    
    lanes = np.asarray(list(set(trajectories["Lane_ID"])))
    
    min_time = min(list(trajectories["Frame_ID"]))
    max_time = max(list(trajectories["Frame_ID"]))
    
    min_loc = min(list(trajectories["Local_Y"]))
    max_loc = max(list(trajectories["Local_Y"]))
    
    times = np.arange(min_time, max_time)
    locs_interval = np.arange(min_loc, max_loc, 100)
    locs = [tuple([loc,loc+100]) for loc in locs_interval]
    # print (locs)
    
    # print (lanes)
    trajectory_lane_location_time = {lane:{loc:{ } for loc in locs} for lane in lanes}
    
    N = len(trajectories)
    
    for i in range(N):
        time = trajectories[i:i+1]["Frame_ID"][i] 
        loc = trajectories[i:i+1]["Local_Y"][i]
        lane = trajectories[i:i+1]["Lane_ID"][i]
        veh = trajectories[i:i+1]["Vehicle_ID"][i]
        speed = trajectories[i:i+1]["v_Vel"][i]
        acc = trajectories[i:i+1]["v_Acc"][i]
        D = {"Time":time, "Location":loc, "Lane":lane, "Speed":speed, "Acceleration":acc}
        
        print ("Importing trajectory #" + str(i))
        # print (trajectory_lane_location_time)
        # print (locs[int((loc-min_loc)/100)])
        
        print (veh in trajectory_lane_location_time[lane][locs[int((loc-min_loc)/100)]].keys())
        
        if veh in trajectory_lane_location_time[lane][locs[int((loc-min_loc)/100)]].keys():
            trajectory_lane_location_time[lane][locs[int((loc-min_loc)/100)]][veh][time] = D
        else:
            trajectory_lane_location_time[lane][locs[int((loc-min_loc)/100)]][veh] = {time: D}
        
    return trajectory_lane_location_time

def get_trajectory_by_loc_time(trajectories_by_lane_loc_veh):      
    trajectories_by_loc_time = {}
    for veh in trajectories_by_lane_loc_veh:
        for time in trajectories_by_lane_loc_veh[veh]:
            if time in trajectories_by_loc_time.keys():
                trajectories_by_loc_time[time][veh] = trajectories_by_lane_loc_veh[veh][time]
            else:
                trajectories_by_loc_time[time] =  { veh: trajectories_by_lane_loc_veh[veh][time] }
    return trajectories_by_loc_time

if __name__ == "__main__":
    path = r"vehicle-trajectory-data\0805am-0820am\trajectories-0805am-0820am.csv"
    us101_data = pd.read_csv(path)
    us101_data_vehicles = list(set(list(us101_data["Vehicle_ID"])))
    us101 = dataset(data = us101_data, vehicles = [])
    spacetime_trj = get_trajectory_by_lane_loc_veh(us101_data)
    # us101_vehicles = us101.data_by_vehicle()
        
import pygad
import random
import numpy as np
L = 4
N= 20
LP = 6
P =  L*LP
solution_initial = [4, 6, 15, 12, 17, 11, 8, 6, 14, 9, 14, 2, 15, 1, 11,8, 11, 10, 14, 8, 14, 8, 9, 15]
solution_Smin=[6, 6, 8, 8, 8, 8,9, 9, 14, 14, 14, 14,11, 11, 11, 15, 15, 15,1, 2, 4, 10, 12, 17] #S_algorithm
solution_Hmin=[1, 2, 4, 6, 6, 8, 8, 8, 8, 9, 9, 10, 11, 11, 11, 12,14, 14, 14, 14, 15, 15, 15, 17]
tf = 5.225 #>> = df/v + v/a + a/j >> where a and j =1
df = 4.2
v =1.6
tdo =2
tdc = 3
tpi =1.2
tpo = 1.2
def fitness_func(solution):
    Total_stops = 0
    Total_h =0
    S_list =[]
    H_list=[]
    for i in range(0,len(solution),LP):
        num_stops = len(set(solution[i:i+LP]))
        Total_stops = Total_stops + num_stops
        S_list.append(num_stops)

    for i in range(0,len(solution),LP):
        h = max(solution[i:i+LP])
        Total_h = Total_h+h
        H_list.append(h)
    #print(S_list)    
    #print(Total_stops) 
    average_h = Total_h/L 
    average_s = Total_stops/L

    #print(H_list)
    #print(average_h) 
     
    RTT = 2*average_h*df/v +(average_s+1)*(tf-df/v+tdo+tdc)+LP*(tpi+tpo)
    #print(H_list)
    #print(average_h)  
    return -1* RTT

print(fitness_func(solution_initial))
print(fitness_func(solution_Smin))
print(fitness_func(solution_Hmin))

import sys
import math
from copy import deepcopy 
import numpy as np

f = open(sys.argv[1])

x = f.readline()
x = x.split()
S = int(x[1])

x = f.readline()
x = x.split()
A = int(x[1])

x = f.readline()
x = x.split()
start = int(x[1])

x = f.readline()
x = x.split()
end = []
end = x[1:]
end = [int(j) for j in end]
if end[0] == -1 :
    end = []

transitions = np.zeros((S,A,S,2))

possible_actions = [[0 for i in range(A)] for j in range(S)]

neighbours = [[[] for i in range(A)] for j in range(S)]

while True :
    x = f.readline()
    x = x.split()
    if x[0] == "transition" :
        t = x[1:]
        t = [float(j) for j in t]
        possible_actions[int(t[0])][int(t[1])] = 1
        neighbours[int(t[0])][int(t[1])].append(int(t[2]))
        transitions[int(t[0])][int(t[1])][int(t[2])][0] = t[3] 
        transitions[int(t[0])][int(t[1])][int(t[2])][1] = t[4]
    else :
        break

discount = float(x[1])

optimal_v = np.zeros(S)
optimal_a = np.zeros(S)
counter = 0

while True :
    old_v = deepcopy(optimal_v)
    for s in range(S) :
        if ( s in end ) :
            continue
        o_v = 0
        for a in range(A) :
            if possible_actions[s][a] == 0 :
                continue
            v = 0
            for d_s in neighbours[s][a] :
                v = v + transitions[s][a][d_s][1]*(transitions[s][a][d_s][0] + discount * optimal_v[d_s])
            # print(v)
            if v > o_v :
                o_v = v
                optimal_a[s] = a
        optimal_v[s] = o_v
    diff_v = abs(old_v - optimal_v)
    counter = counter + 1
    # print(counter)
    if (max(diff_v)) < 1e-16 :
        # print(counter)
        break

for s in end :
    optimal_a[s] = -1

for s in range(S) :
    print(str(optimal_v[s]) + " " + str(int(optimal_a[s])))


print("iterations "+str(counter))
import sys
import math
import numpy as np

maze = np.loadtxt(sys.argv[1], dtype='i')
# print(maze)

unique, counts = np.unique(maze, return_counts=True)
d = dict(zip(unique, counts))

S = d[0] + 2
A = 4
start = 0
end = 0
transitions = np.zeros((S,A,S,2))

state = 0
shape = np.shape(maze)
maze = maze.reshape(-1)

for i in range(len(maze)) :
	x = maze[i]
	if x == 1 :
		continue
	elif x == 0 :
		maze[i] = state+4
	elif x == 2 :
		start = state
		maze[i] = state+4
	elif x == 3 :
		end = state
		maze[i] = state+4
	state = state + 1

maze = maze.reshape(shape)

lll = [ [] for j in range(S) ]

for i in range(len(maze)) :
	for j in range(len(maze[0])) :
		x = maze[i][j]

		if x == 1 :
			continue

		up = maze[i][j-1]-4
		down = maze[i][j+1]-4
		left = maze[i-1][j]-4
		right = maze[i+1][j]-4
		x = x-4

		lll[x] = [up,down,left,right]

		if up >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][0][up][1] = 1
			transitions[x][0][up][0] = 1.0*(up == end)

		if down >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][1][down][1] = 1
			transitions[x][1][down][0] = 1.0*(down == end)

		if left >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][2][left][1] = 1
			transitions[x][2][left][0] = 1.0*(left == end)

		if right >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][3][right][1] = 1
			transitions[x][3][right][0] = 1.0*(right == end)

f = open(sys.argv[2],'r')

optimal_policy = []

while True :
	x = f.readline()
	x = x.split()
	# print(x)
	if x[0] == "iterations" :
		break
	else :
		optimal_policy.append(int(x[1]))

# print(optimal_policy)

moves = []

s = int(start)

while s != end :
	# print(s)
	s = int(s)
	if optimal_policy[s] == 0 :
		moves.append("W")
		s = lll[s][0]
	elif optimal_policy[s] == 1 :
		moves.append("E")
		s = lll[s][1]
	elif optimal_policy[s] == 2 :
		moves.append("N")
		s = lll[s][2]
	elif optimal_policy[s] == 3 :
		moves.append("S")
		s = lll[s][3]

print(" ".join(moves))

	


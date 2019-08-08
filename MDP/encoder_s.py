import sys
import math
import numpy as np

maze = np.loadtxt(sys.argv[1], dtype='i')
# print(maze)

p = float(sys.argv[2])

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


print("numStates "+str(S))
print("numActions "+str(A))
print("start "+str(start))
print("end "+str(end))

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

		movez = 0

		if up >= 0 :
			movez = movez + 1
		if down >= 0 :
			movez = movez + 1
		if left >= 0 :
			movez = movez + 1
		if right >=0 :
			movez = movez + 1

		# print(x,lll[x])

		if up >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][0][up][1] = p + (1-p)/movez
			transitions[x][0][up][0] = 1.0*(up == end)

		if down >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][0][down][1] = (1-p)/movez
			transitions[x][0][down][0] = 1.0*(down == end)

		if left >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][0][left][1] = (1-p)/movez
			transitions[x][0][left][0] = 1.0*(left == end)

		if right >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][0][right][1] = (1-p)/movez
			transitions[x][0][right][0] = 1.0*(right == end)


		if down >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][1][down][1] = p + (1-p)/movez
			transitions[x][1][down][0] = 1.0*(down == end)

		if up >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][1][up][1] = (1-p)/movez
			transitions[x][1][up][0] = 1.0*(up == end)

		if left >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][1][left][1] = (1-p)/movez
			transitions[x][1][left][0] = 1.0*(left == end)

		if right >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][1][right][1] = (1-p)/movez
			transitions[x][1][right][0] = 1.0*(right == end)

		if left >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][2][left][1] = p + (1-p)/movez
			transitions[x][2][left][0] = 1.0*(left == end)

		if down >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][2][down][1] = (1-p)/movez
			transitions[x][2][down][0] = 1.0*(down == end)

		if up >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][2][up][1] = (1-p)/movez
			transitions[x][2][up][0] = 1.0*(up == end)

		if right >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][2][right][1] = (1-p)/movez
			transitions[x][2][right][0] = 1.0*(right == end)

		if right >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][3][right][1] = p + (1-p)/movez
			transitions[x][3][right][0] = 1.0*(right == end)

		if down >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][3][down][1] = (1-p)/movez
			transitions[x][3][down][0] = 1.0*(down == end)

		if left >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][3][left][1] = (1-p)/movez
			transitions[x][3][left][0] = 1.0*(left == end)

		if up >= 0 :                                                     # 0 -> up 1 -> down 2 -> left 3 -> right
			transitions[x][3][up][1] = (1-p)/movez
			transitions[x][3][up][0] = 1.0*(up == end)
		
# print(lll)

for s in range(S) :
	for a in range(A) :
		for d_s in lll[s] :
			if d_s < 0 :
				continue
			elif transitions[s][a][d_s][1] == 0 :
				continue
			else :
				print( "transition "+str(s)+" "+str(a)+" "+str(d_s)+" "+str(transitions[s][a][d_s][0])+" "+str(transitions[s][a][d_s][1]) )
			

print("discount 0.9")

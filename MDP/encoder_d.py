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
		
# print(lll)

for s in range(S) :
	for a in range(A) :
		d_s = lll[s][a]
		if d_s < 0 :
			continue
		else :
			print( "transition "+str(s)+" "+str(a)+" "+str(d_s)+" "+str(transitions[s][a][d_s][0])+" "+str(transitions[s][a][d_s][1]) )
			

print("discount 0.9")

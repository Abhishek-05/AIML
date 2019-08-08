import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		y = np.matmul(X,self.weights)
		y += self.biases
		y = sigmoid(y)
		self.data = y
		return y
		raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# y = np.matmul(activation_prev,self.weights)
		# y += self.biases
		# y = sigmoid(y)
		y = self.data
		y = y * (1-y)
		z = delta * y
		b = z.sum()
		w = np.matmul(np.transpose(activation_prev),z)
		r = np.matmul(z,np.transpose(self.weights))
		self.biases -= lr * b
		self.weights -= lr * w
		return r
		raise NotImplementedError
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		y = np.zeros((n,self.out_depth,self.out_row,self.out_col))
		for N in range(n) :
			for c in range(self.out_depth):
				w1 = self.weights[c]
				for a in range(self.out_row) :
					for b in range(self.out_col) :
						i = a*self.stride
						j = b*self.stride
						# for m in range(self.filter_row) :
						# 	for n in range(self.filter_col) :
						# 		for o in range(self.in_depth) :
						# 			w = w1[o]
									# w = np.flipud(w)
									# w = np.fliplr(w)
									#y[N][c][a][b] += X[N][o][i+m][j+n]*w[m][n]
						y[N][c][a][b] += (X[N][0:self.in_depth, i:i+self.filter_row, j:j+self.filter_col]*w1).sum()
				y[N][c] += self.biases[c]
		self.data = sigmoid(y)
		return self.data


						
		raise NotImplementedError
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n2 = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		#print("Started backward pass")
		g_dash = self.data * (1-self.data)
		b_e1 = delta * g_dash
		b_e = np.sum(b_e1,axis=0)
		b = b_e.sum(axis=1).sum(axis=1)
		self.biases -= lr * b
		#print("updated biases")
		r = np.zeros((n2 ,self.in_depth ,self.in_row ,self.in_col))

		for nn in range(n2) :
			for dd in range(self.out_depth) :
				for i in range(self.out_row) :
					for j in range(self.out_col) :
						a = i * self.stride
						b = j * self.stride
						# for d in range(self.in_depth) :
						# 	for m in range(self.filter_row) :
						# 		for n in range(self.filter_col) :
									#r[nn][d][a+m][b+n] += b_e1[nn][dd][i][j]*self.weights[dd][d][m][n]
						r[nn][0:self.in_depth, a:a+self.filter_row, b:b+self.filter_col]+=b_e1[nn][dd][i][j]*self.weights[dd]
		#print("found out r")
		w = np.zeros((self.out_depth ,self.in_depth ,self.filter_row ,self.filter_col))

		for nn in range(n2) :
			for dd in range(self.out_depth) :
				for i in range(self.out_row) :
					for j in range(self.out_col) :
						a = i * self.stride
						b = j * self.stride
						# for d in range(self.in_depth) :
						# 	for m in range(self.filter_row) :
						# 		for n in range(self.filter_col) :							
						x = b_e1[nn][dd][i][j]
						# 			y = activation_prev[nn][d][i*self.stride + m][j*self.stride + n]
						y = activation_prev[nn][0:self.in_depth, a:a+self.filter_row, b:b+self.filter_col]
									# w[dd][d][m][n] += x*y
						w[dd] += x*y 


		self.weights -= lr * w
		# print(self.weights)
		return r
		raise NotImplementedError
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		y = np.zeros((n,self.out_depth,self.out_row,self.out_col))
		for N in range(n) :
			for c in range(self.out_depth):
				for a in range(self.out_row) :
					for b in range(self.out_col) :
						i = a*self.stride
						j = b*self.stride
						# for m in range(self.filter_row) :
						# 	for n in range(self.filter_col) :
						y[N][c][a][b] += X[N][c][i:i+self.filter_row, j:j+self.filter_col].sum()
			#y[N] += self.biases.T
			#y = sigmoid(y)
		y = y/(self.filter_col*self.filter_row)
		return y

		raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n2 = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		r = np.zeros((n2 ,self.in_depth ,self.in_row ,self.in_col))

		for nn in range(n2) :
			for dd in range(self.out_depth) :
				for i in range(self.out_row) :
					for j in range(self.out_col) :
						a = i * self.stride
						b = j * self.stride
						r[nn][0:self.in_depth, a:a+self.filter_row, b:b+self.filter_col] += delta[nn][dd][i][j]

		return r
		raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

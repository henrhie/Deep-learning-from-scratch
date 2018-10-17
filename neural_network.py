import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import math

def main():
	data = pd.read_csv("c:/users/henry/desktop/datasets/b_cancer.csv")  #loads dataset
	lb = LabelEncoder()    #instantiate the LabelEncoder class
	x = data.iloc[:,2:32].values
	y = data["diagnosis"].values
	y = lb.fit_transform(y) #converts the labels into ones and zeros
	x = (x-np.mean(x))/np.std(x)  #normalize the input features
	x_train,x_test,y_train,y_test = train_test_split(x,y , test_size=0.3, random_state=0)
	x_train = x_train.T   #transposes the dataset to make matrix multiplication feasible
	x_test = x_test.T
	y_train = y_train.reshape(-1,1)
	y_train = y_train.T
	y_test = y_test.reshape(-1,1)
	y_test = y_test.T
	layer_dims =[x_train.shape[0],42,62,12,20,11,1]
	learning_rate=1e-3
	print("learning_rate for training is " + str(learning_rate))
	print(y_test)
	parameters = network_model(x_train,y_train,x_test,y_test,learning_rate=learning_rate,epochs=10000,layer_dims=layer_dims,lambd=0.0,learning_decay=0.00000001,p_keep=1.0,beta=0.9,optimizer="gradient descent")
	train_predictions = predict(x_train,parameters)
	predictions = predict(x_test,parameters)
	print(train_predictions)
	print(predictions)
	train_score = accuracy_score(train_predictions,y_train)
	print(train_score)
	score = accuracy_score(predictions,y_test)
	print(score)

def initialize_parameters(layer_dims):
	parameters = {}
	L = len(layer_dims)
	for l in range(1,L):
		parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * np.sqrt(2/(layer_dims[l-1])) # He weight initialization technique..By He et Al
		parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
	return parameters

def linear_forward(A,W,b):
	Z = np.dot(W,A) + b
	cache = (A,W,b)
	return Z,cache

def sigmoid(Z):
	cache = Z
	s = 1/(1 + np.exp(-Z))
	return s, cache

def relu(Z):
	s =np.maximum(0,Z)
	cache = Z
	return s,cache

def leaky_relu(Z, alpha):
	s = np.maximum(Z*alpha,Z)
	cache = Z
	return s,cache

def linear_activation_forward(A_prev,W,b,activation):
	if activation == "relu":
		Z,linear_cache = linear_forward(A_prev,W,b)
		A, activation_cache = relu(Z)

	elif activation =="sigmoid":
		Z,linear_cache = linear_forward(A_prev,W,b)
		A, activation_cache = sigmoid(Z)
	cache = (linear_cache,activation_cache)
	return A, cache

def L_model_forward(X, parameters,p_keep=1):
	caches = []
	dropout_dict={}
	L = len(parameters) //2
	A = X
	for i in range(1,L):
		A_prev = A
		A,cache = linear_activation_forward(A_prev,parameters["W" + str(i)],parameters["b" + str(i)], activation="relu")
		dropout_dict["D"+ str(i)]= np.random.rand(A.shape[0],A.shape[1])
		dropout_dict["D" + str(i)] = dropout_dict["D" + str(i)] < p_keep
		A = A*dropout_dict["D" + str(i)]
		A/=p_keep
		caches.append(cache)
	AL, cache = linear_activation_forward(A,parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
	caches.append(cache)
	return AL, caches,dropout_dict
def relu_backward(dA,Z):
	A,_ = relu(Z)
	s = (A>0)
	dZ = dA * s
	return dZ

def sigmoid_backward(dA,Z):
	s,cache = sigmoid(Z)
	derivative = s * (1-s)
	dZ = dA * derivative
	return dZ

def linear_backward(dZ, cache,lambd):
	m = len(cache)
	linear_cache, activation_cache = cache
	A_prev,W,b = linear_cache
	Z = activation_cache
	dW = 1/m *(np.dot(dZ,A_prev.T) + (lambd*W))
	db = 1/m * np.sum(dZ, axis=1, keepdims = True)
	dA_prev = np.dot(W.T, dZ)
	return dW, db,dA_prev

def linear_backward_activation(dA, cache, activation,lambd):
	if activation == "relu":
		linear_cache,activation_cache = cache
		Z = activation_cache
		dZ  = relu_backward(dA,Z)
		dW,db,dA_prev = linear_backward(dZ, cache,lambd)
	elif activation == "sigmoid":
		linear_cache,activation_cache = cache
		Z = activation_cache
		dZ = sigmoid_backward(dA,Z)
		dW,db,dA_prev = linear_backward(dZ, cache,lambd)
	return dW, db, dA_prev

def l_model_backward(AL,Y,cache,lambd,dropout_dict,p_keep):
	grads = {}
	Y.shape = (AL.shape)
	dAL = -np.divide(Y,AL) + np.divide(1-Y,1-AL+(1e-18))
	current_cache = cache[-1]
	L = len(cache)
	grads["dW" + str(L)],grads["db" + str(L)], grads["dA" + str(L-1)] = linear_backward_activation(dAL, current_cache, activation ="sigmoid",lambd=0.0)
	grads["dA"+ str(L-1)] = grads["dA" + str(L-1)] *dropout_dict["D" + str(L-1)]
	grads["dA" + str(L-1)]/=p_keep
	for i in reversed(range(L-1)):
		current_cache = cache[i]
		grads["dW"+ str(i+1)], grads["db" + str(i+1)], grads["dA"+ str(i)] = linear_backward_activation(grads["dA" + str(i+1)],current_cache,activation="relu",lambd=0.0)
		if i == 0:
			break
		else:
			grads["dA"+ str(i)] = grads["dA" + str(i)] * dropout_dict["D" + str(i)]
			grads["dA" + str(i)] /=p_keep
	return grads

def update_parameters(parameters,grads, learning_rate):
	L = len(parameters) //2
	for l  in range(1,L):
		parameters["W"+ str(l)] =parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
		parameters["b" + str(l)] = parameters["b"+ str(l)] - learning_rate * grads["db" + str(l)]
	return parameters

def dict_to_vector(dictionary):
	values = []
	keys = []
	for key,value in dictionary.items():
		values.append(value)
		keys.append(key)
	new_vector = np.array(values)
	new_vector = new_vector.reshape(-1,1)
	new_keys = np.array(keys)
	return new_vector, new_keys

def vector_to_dict(vector,keys):
    dict={}
    for i in range(len(keys)):
        dict[keys[i]]  = vector[i]
    return dictionary

def extract_weight(dict):
    L = len(dict)//2
    values=[]
    for i in range(1,L+1):
        values.append(dict["W" + str(i)])
    return values

def calc_norm(weight):
    norm =0
    L = len(weight)
    for i in range(L):
        norm+=np.sum(np.square(weight[i]))
    return norm

def random_mini_batches(X,Y,mini_batch_size,seed=0):
	mini_batch =[]
	m = Y.shape[1]
	permutation = list(np.random.permutation(m))
	X_shuffled = X[:,permutation]
	Y_shuffled = Y[:,permutation].reshape(Y.shape[0],m)
	num_complete_minibatches = math.floor(m/mini_batch_size)
	for i in range(num_complete_minibatches):
		X_minibatch = X_shuffled[:,mini_batch_size*i:mini_batch_size*(i+1)]
		Y_minibatch = Y_shuffled[:,mini_batch_size*i:mini_batch_size*(i+1)]
		minibatches = (X_minibatch,Y_minibatch)
		mini_batch.append(minibatches)
	if m % mini_batch_size !=0:
		end = m - mini_batch_size * math.floor(m / mini_batch_size)
		X_minibatch = X_shuffled[:,num_complete_minibatches*mini_batch_size:]
		Y_minibatch = Y_shuffled[:,num_complete_minibatches*mini_batch_size:]
		minibatches = (X_minibatch,Y_minibatch)
		mini_batch.append(minibatches)
	return mini_batch

def initialize_velocities(params):
	v ={}
	L = len(params)//2
	for i in range(L):
		v["dW" + str(i+1)] = np.zeros_like(params["W" + str(i+1)])
		v["db" + str(i+1)] = np.zeros_like(params["b" + str(i+1)])
	return v


def update_parameters_with_momentum(params,learning_rate,grads,v,beta):
	L = len(params)//2
	for i in range(L):
		v["dW" + str(i+1)] = beta*v["dW" + str(i+1)] + (1-beta)*grads["dW" + str(i+1)]
		v["db" + str(i+1)] = beta*v["db" + str(i+1)] + (1-beta)*grads["db" + str(i+1)]

		params["W"+ str(i+1)] = params["W" + str(i+1)] - learning_rate*v["dW"+ str(i+1)]
		params["b" + str(i+1)] = params["b" + str(i+1)] - learning_rate*v["db" + str(i+1)]
	return params,v

def initialize_rmsprop(params):
	L = len(params)//2
	s={}
	for l in range(L):
		s["dW" + str(l+1)] = np.zeros_like(params["W" +  str(l+1)])
		s["db" + str(l+1)] = np.zeros_like(params["W"+ str((l+1))])
	return s

def update_rmsprop(s,t,params,grads,learning_rate,beta_2=0.999,epsilon=1e-8):
	L = len(grads)//2
	s_corrected ={}
	for l in range(L):
		s["dW" + str(l+1)] = (s["dW"+ str(l+1)]*beta2) + (1-beta2) * np.square(grads["dW" + str(l+1)])
		s["db" + str(l+1)] = (s["db"+ str(l+1)]*beta2) + (1-beta2) * np.square(grads["db" + str(l+1)])
		s_corrected["dW" + str(l+1)] = np.divide(s["dW" + str(l+1)],1 - np.power(beta2,t))
		s_corrected["db" + str(l+1)] = np.divide(s["db" + str(l+1)],1 - np.power(beta2,t))
		params["W" + str(l+1)] = params["W" + str(l+1)] - np.divide(learning_rate,np.sqrt(s_corrected["dW" + str(l+1)]  + epsilon))
		params["b" + str(l+1)] = params["b" + str(l+1)] - np.divide(learning_rate,np.sqrt(s_corrected["db" + str(l+1)] + epsilon))
	return params,s_corrected

def initialize_adam(params):
	L = len(params)//2
	s={}
	v={}
	for l in range(L):
		v["dW" + str(l+1)] = np.zeros_like(params["W" + str(l+1)])
		v["db" + str(l+1)] = np.zeros_like(params["b" + str(l+1)])
		s["dW" + str(l+1)] = np.zeros_like(params["W" + str(l+1)])
		s["db"+ str(l+1)] = np.zeros_like(params["b" + str(l+1)])
	return v,s

def update_adam(params,grads,v,s,t,learning_rate,epsilon=1e-8,beta1=0.9,beta2=0.999):
	v_corrected= {}
	s_corrected ={}
	L =len(params)//2
	for l in range(L):
		v["dW" + str(l+1)] = v["dW" + str(l+1)]*beta1 + (1-beta1)*grads["dW" + str(l+1)]
		v["db" + str(l+1)] = v["db" + str(l+1)]*beta1 + (1-beta1)*grads["db" + str(l+1)]
		v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-np.power(beta1,t))
		v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-np.power(beta1,t))
		s["dW" + str(l+1)] = s["dW" + str(l+1)]*beta2 + (1-beta2)*(grads["dW" + str(l+1)])**2
		s["db" + str(l+1)] = s["db" + str(l+1)]*beta2 + (1-beta2)*(grads["db" + str(l+1)])**2
		s_corrected["dW" + str(l+1)] = (s["dW" + str(l+1)])/(1 - np.power(beta2,t))
		s_corrected["db" + str(l+1)] = (s["db" + str(l+1)])/(1 - np.power(beta2,t))
		params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate*np.divide(v_corrected["dW" + str(l+1)],np.sqrt(s_corrected["dW" + str(l+1)]+epsilon))
		params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate*np.divide(v_corrected["db" + str(l+1)],np.sqrt(s_corrected["db" + str(l+1)]+epsilon))
	return params,s_corrected,v_corrected

def compute_cost(AL,Y,lambd,parameters):
	L = len(parameters)//2
	m = AL.shape[1]
	weight_array = extract_weight(parameters)
	norm = calc_norm(weight_array)
	regu_term =1/m*(lambd/2) * norm # l2 regularization term
	cost_intial = -1/m * np.sum((Y*np.log(AL)) + (1-Y)*(np.log(1-AL)))
	cost = cost_intial + regu_term
	return cost


def network_model(x_train,y_train,x_test,y_test,learning_rate,epochs,layer_dims,lambd,learning_decay,p_keep,beta,optimizer = None):
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	parameters = initialize_parameters(layer_dims)
	t=0
	costs = []
	cost1=[]
	scores1 = []
	scores2 =[]
	v_adam,s_adam = initialize_adam(parameters)
	v_momentum = initialize_velocities(parameters)
	s_prop = initialize_rmsprop(parameters)
	for i in range(epochs):
		learning_rate = learning_rate - (learning_rate*learning_decay)
		minibatches = random_mini_batches(x_train,y_train,mini_batch_size=16)
		for mini_batch in minibatches:
			X_minibatch,Y_minibatch = mini_batch
			AL,cache,dropout_dict = L_model_forward(X_minibatch,parameters,p_keep)
			cost_train = compute_cost(AL,Y_minibatch,lambd,parameters)
			costs.append(cost_train)
			grads = l_model_backward(AL,Y_minibatch, cache,lambd,dropout_dict,p_keep)
			if optimizer is not None:

				if optimizer == "gradient descent": #Gradient Descent
					parameters = update_parameters(parameters,grads,learning_rate)

				elif optimizer == "adam": # Adaptive Moment Estimation Adam
					t+=1
					parameters,s_adam,v_adam = update_adam(parameters,grads,v_adam,s_adam,t,learning_rate)

				elif optimizer == "gradient descent with momentum": #Gradient Descent with momentum
					parameters,v_momentum = update_parameters_with_momentum(parameters,learning_rate,grads,v_momentum,beta)

				elif optimzer == "rmsprop":
					t+=1
					parameters,s_prop = update_rmsprop(s_prop,t,parameters,grads,learning_rate)


			predictions = predict(x_train,parameters)
			score = accuracy_score(predictions,y_train)
			scores1.append(score)
		if i%50 ==0:
			print("cross entropy loss after "+ str(i) + "th epoch = " + str(cost_train))
			print("accuracy after " + str(i) + "th epoch = " + str(score))
			print("current learning_rate = " + str(learning_rate))
	ax1.plot(costs)
	ax2.plot(scores1, label = " training set")
	plt.legend()
	plt.show()
	return parameters

def predict(x_test,parameters):
	predictions,_,_ = L_model_forward(x_test,parameters)
	for i in range(predictions.shape[1]):
		if predictions[0,i] >= 0.5:
			predictions[0,i] = 1
		elif predictions[0,i] < 0.5:
			predictions[0,i] = 0
	return predictions

def accuracy_score(predictions,actual):
	counter = 0
	for i in range(predictions.shape[1]):
		if predictions[0,i] == actual[0,i]:
			counter+=1
		else:
			pass
	return counter/predictions.shape[1]

if __name__ == '__main__':
	main()

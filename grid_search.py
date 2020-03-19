import time
start = time.time()

import csv
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics
# from sklearn.preprocessing import OneHotEncoder
# from tensorflow import keras
# from keras.utils import to_categorical
# from keras import metrics
# from keras import metrics


def get_data(filename):# gets the value and labels from the csv file 
	with open(filename) as f:
		reader = csv.reader(f,delimiter=',')
		# file = np.array(list(reader)).astype('int')
		file = np.array(list(reader)).astype('float')
		# print(file[:,-1].squeeze())
		labels = np.asarray(file[:,-1].squeeze())
		labels = labels.astype(int)
		data = file[:,:-1]
		# f.close()#No need to do this
	return data,labels

def sigmoid(z):
	s = 1/(1+np.exp(-z))
	return s
def sigmoid_prime(x):
    """ x is a numpy array of numbers. Returns an array which
	    contains sig'[z] for all elements z in x"""
    return sigmoid(x)*(1 - sigmoid(x))

def softmax(x):
	"""x is a vector of activations. Returns a vector which is
    softmax(x)"""
	exp_vec = np.exp(x)
	s = np.sum(exp_vec,axis = 0)
	try:	
		if s != 0:
			softmax = exp_vec/s
			return softmax
		else:
			return np.zeros(exp_vec.shape)
	except:
		return np.zeros(exp_vec.shape)

def relu(z):
	return np.maximum(0,z)
def relu_prime(x):
    """ returns the derivative of the relu function"""
    return 0.5*np.sign(x) + 0.5

def act_funcn_update(ip,fid):
	'''
	ip == input
	fid == function id, for changing the activation function associated with a layer.
	1 - sigmoid (default)
	2 - relu
	3 - softmax
	4 - tan sigmoid
	'''
	if fid == 2:
		op = relu(ip)
		op_prime = relu_prime(ip)
		return op, op_prime
	elif fid ==3:
		op = softmax(ip)
		return op, op
	elif fid == 1:
		op = sigmoid(ip)
		op_prime = sigmoid_prime(ip)
		return op, op_prime
	elif fid ==4:
		op = np.tanh(ip)
		op_prime = 1 - op**2
		return op, op_prime
	else:
		op = sigmoid(ip)
		op_prime = sigmoid_prime(ip)
		return op, op_prime

def d_act_funcn(ip,fid):
	'''
	ip == input
	fid == function id, for changing the activation function associated with a layer.
	1 - sigmoid (default)
	2 - relu
	3 - softmax
	4 - tanh
	'''
	if fid == 2:
		op = relu(ip)
		op_prime = relu_prime(ip)
		return op_prime
	elif fid ==3:
		op = softmax(ip)
		return op
	elif fid == 1:
		# op = sigmoid(ip)
		op_prime = sigmoid_prime(ip)
		return op_prime
	elif fid ==4:
		op = np.tanh(ip)
		op_prime = 1 - op**2
		return op_prime
	else:
		# op = sigmoid(ip)
		op_prime = sigmoid_prime(ip)
		return op_prime

def act_funcn(ip,fid):
	'''
	ip == input
	fid == function id, for changing the activation function associated with a layer.
	1 - sigmoid (default)
	2 - relu
	3 - softmax
	4 - tanh
	'''
	if fid == 2:
		op = relu(ip)
		return op
	elif fid ==3:
		op = softmax(ip)
		return op
	elif fid == 1:
		op = sigmoid(ip)
		return op
	elif fid ==4:
		op = np.tanh(ip)
		# op_prime = 1 - op**2
		return op
	else:
		op = sigmoid(ip)
		return op



class neuralNetwork:
	def __init__(self,layer_sizes,f_ids):
		self.layer_sizes = layer_sizes
		self.num_layers = len(layer_sizes)
		weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
		self.weights =  [0.1*np.random.standard_normal(s) for s in weight_shapes]
		self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]
		self.act_funcn_list = f_ids

	def predict(self,a):
		for w,b,f in zip(self.weights,self.biases,self.act_funcn_list):
			a = act_funcn(np.dot(w,a) + b,f)
		return a
	def predict_batch(self,a):
		a = a.T
		for w,b,f in zip(self.weights,self.biases,self.act_funcn_list):
			a = act_funcn(np.dot(w,a) + b,f)
		return a.T

	def cost(self,pred,target):
		pass

	def d_cost(self,pred,target):
		temp=0
		for w in self.weights:
			# print(temp)
			temp += np.linalg.norm(w) 
		return pred - target + self.lamb*temp

	def backprop1(self,inp,targets):
		''' for multiple samples''' # currently incomplete
		# batch_size = inp.shape[1]
		layer_ip = []
		inp = inp.reshape((inp.shape[0],1))
		targets = targets.reshape((targets.shape[0],1))
		d_layer_ip = []
		d_layer_ip.append(None)
		layer_ip.append(inp)
		for w,b,f in zip(self.weights,self.biases,self.act_funcn_list):
			ip,ip_prime = act_funcn_update(np.matmul(w,layer_ip[-1]) + b,f)
			layer_ip.append(ip)
			d_layer_ip.append(ip_prime)
			# print(layer_ip[-1].shape,d_layer_ip[-1].shape)

		# d_layer_ip[-1] /= batch_size

		# print(layer_ip[-1].shape,  targets.shape, d_layer_ip[-1].shape)
		err = self.d_cost(layer_ip[-1] , targets)* d_layer_ip[-1] 
		# print(err.shape)
		dw = []
		db = []
		db.append(err)
		dw.append(np.dot(err,layer_ip[-2].T))
		for i in range(2,(self.num_layers)):
			# print(self.weights[-i+1].T.shape,err.shape,d_layer_ip[-i].shape)
			err = np.dot(self.weights[-i+1].T,err)*d_layer_ip[-i]
			# print(err.shape)
			db.insert(0,err)
			dw.insert(0,np.dot(err,layer_ip[-i-1].T))
		pass
		# nabla_w = [np.zeros(w.shape) for w in self.weights]
		# nabla_b = [np.zerod(b.shape) for b in self.biases]
		return dw,db
		
	def backprop(self,x,y):
		''' for single sample at a time'''
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]

		x = x.reshape(x.shape[0],1)
		y = y.reshape(y.shape[0],1)

		activation = x
		activations = [x] #storing layer by layer activation
		zs =[] # storing the inputs before activation function in each layer
		#feedforward
		for w,b,f in zip(self.weights,self.biases,self.act_funcn_list):
			# print(w.shape, activation.shape,b.shape)
			z = np.dot(w,activation) + b
			zs.append(z)
			activation = act_funcn(z,f)
			activations.append(activation)
		#backprop
		# print(activations[-1].shape,y.shape)
		delta = self.d_cost(activations[-1],y) * d_act_funcn(activations[-1],self.act_funcn_list[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta,activations[-2].T)
		for i in range(2,len(self.layer_sizes)):
			z = zs[-i]
			sp = d_act_funcn(z,self.act_funcn_list[-i])
			delta = np.dot(self.weights[-i+1].T,delta)*sp
			nabla_b[-i] = delta
			nabla_w[-i] = np.dot(delta, activations[-i-1].T)
		# print([b.shape for b in nabla_b])
		return nabla_w,nabla_b

	def update_batch(self, x_batch, y_batch, lr):
	    l = x_batch.shape[0]
	    # print(l)
	    nabla_b = [np.zeros(b.shape) for b in self.biases]
	    nabla_w = [np.zeros(w.shape) for w in self.weights]
	    for x, y in zip(x_batch,y_batch):
	        # print(x.shape,y.shape)
	        delta_nabla_w, delta_nabla_b = self.backprop(x, y)
	        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
	        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
	    # self.weights = [ print(w.shape,nw.shape) 
	    # print([(w.shape,nw.shape) for w, nw in zip(self.weights, nabla_w)])
	    # print([(b.shape,nb.shape) for b, nb in zip(self.biases, nabla_b)])
	    self.weights = [w-(lr/l)*nw 
	                    for w, nw in zip(self.weights, nabla_w)]
	    self.biases = [b-(lr/l)*nb 
	                   for b, nb in zip(self.biases, nabla_b)]
	    return

	def train(self,xtrain,ytrain,epochs,lr,c, batch_size ):
		self.epochs=epochs
		self.lr=lr
		self.lamb=c
		self.batch_size=batch_size
		for e in range(epochs):
			i =0
			while i < len(ytrain):
				xbatch = xtrain[i:i+batch_size,:]
				ybatch = ytrain[i:i+batch_size,:]
				N = xbatch.shape[0]
				i += batch_size
				self.update_batch(xbatch,ybatch,lr)
		return


filename = '2017EE10458.csv'
data,labels = get_data(filename)
# onehot_encoder = OneHotEncoder(sparse=False)
# targets = onehot_encoder.fit_transform(labels)
data = data/255.0
targets = to_categorical(labels)
# targets = np.zeros((labels.size, labels.max()+1))
# targets[np.arange(labels.size,labels)]=1
# targets = labels
x_train, x_test,y_train,y_test = train_test_split(data,targets, train_size=0.75,test_size=0.25,random_state=12)


layer_set0 = [(784,10)]
# layer_set0 = [(25,10)]
layer_set1 = [(784,25,10),(784,400,10)]
# layer_set1 = [(784,400,10)]
# layer_set1 = [(25,16,10)]
# layer_set2 = [(784,400,25,10)]
layer_set2 = [(25,16,16,10)]
layer_set3 = [(784,400,100,25,10)]

non_lin_set0 = [(1,0),(2,0),(3,0),(4,0)]
non_lin_set1 = [(2,2),(4,4)]
# non_lin_set1 = [(4,4)]
non_lin_set2 = [(2,2,2),(4,4,4)]
non_lin_set3 = [(4,2,2,2)]

lambda_set = [1e-9,1e-7,1e-5,1e-3,1e-1,1,1e1]
lr_set = [1e-3,1e-1,0.5,1,1.5,1e1,1e2]
batch_set = [1,8,16,32,256]
epochs = 15

y_test_1 = y_test.argmax(1)
y_train_1 = y_train.argmax(1)
acc = metrics.Accuracy()
lamb = 0

csv.register_dialect('myDialect', delimiter = ';')
with open('graddes_data.txt', 'w') as writeFile:
	writer = csv.writer(writeFile, dialect="myDialect")
	# writer.writerows(lines)
	
	for layers in layer_set0:
		for fids in non_lin_set0: 
			for lamb in lambda_set:
				for lr in lr_set:
					for batch_size in batch_set:
						t1 = time.time()
						a = neuralNetwork(layers,fids)
						a.train(x_train,y_train,epochs,lr,lamb,batch_size)
						y_pred = a.predict_batch(x_train)
						y_pred = y_pred.argmax(1)
						acc.update_state(y_train_1,y_pred)
						train_acc = acc.result().numpy()

						y_pred = a.predict_batch(x_test)
						y_pred = y_pred.argmax(1)
						acc.update_state(y_test_1,y_pred)
						test_acc = acc.result().numpy()
						t2 = time.time() - t1
						strng = str(layers)+';'+str(fids)+';'+str(lr)+';'+str(lamb)+';'+str(batch_size) +';'+str(epochs) +';train; '+str(train_acc) +';test ;'+str(test_acc) + ';time;'+str(t2) + '\n'
						print(strng)
						writeFile.write(strng)
					# writeFile.close()
					# print('hello')

	for layers in layer_set1:
		for fids in non_lin_set1: 
			for lamb in lambda_set:
				for lr in lr_set:
					for batch_size in batch_set:
						t1 = time.time()
						a = neuralNetwork(layers,fids)
						a.train(x_train,y_train,epochs,lr,lamb,batch_size)
						y_pred = a.predict_batch(x_train)
						y_pred = y_pred.argmax(1)
						acc.update_state(y_train_1,y_pred)
						train_acc = acc.result().numpy()

						y_pred = a.predict_batch(x_test)
						y_pred = y_pred.argmax(1)
						acc.update_state(y_test_1,y_pred)
						test_acc = acc.result().numpy()
						t2 = time.time() - t1
						strng = str(layers)+';'+str(fids)+';'+str(lr)+';'+str(lamb)+';'+str(batch_size) +';'+str(epochs) +';train; '+str(train_acc) +';test ;'+str(test_acc) + ';time;'+str(t2) + '\n'
						print(strng)
						writeFile.write(strng)					
					# print(layers,fids,lr,batch_size,epochs ,'train: ',train_acc,'\ttest :',test_acc, 'time:',t2)

	for layers in layer_set2:
		for fids in non_lin_set2: 
			for lamb in lambda_set:
				for lr in lr_set:
					for batch_size in batch_set:
						t1 = time.time()
						a = neuralNetwork(layers,fids)
						a.train(x_train,y_train,epochs,lr,lamb,batch_size)
						y_pred = a.predict_batch(x_train)
						y_pred = y_pred.argmax(1)
						acc.update_state(y_train_1,y_pred)
						train_acc = acc.result().numpy()

						y_pred = a.predict_batch(x_test)
						y_pred = y_pred.argmax(1)
						acc.update_state(y_test_1,y_pred)
						test_acc = acc.result().numpy()
						t2 = time.time() - t1
						strng = str(layers)+';'+str(fids)+';'+str(lr)+';'+str(lamb)+';'+str(batch_size) +';'+str(epochs) +';train; '+str(train_acc) +';test ;'+str(test_acc) + ';time;'+str(t2) + '\n'
						print(strng)
						writeFile.write(strng)					
					# print(layers,fids,lr,batch_size,epochs ,'train: ',train_acc,'\ttest :',test_acc, 'time:',t2)


	for layers in layer_set3:
		for fids in non_lin_set3: 
			for lamb in lambda_set:
				for lr in lr_set:
					for batch_size in batch_set:
						t1 = time.time()
						a = neuralNetwork(layers,fids)
						a.train(x_train,y_train,epochs,lr,lamb,batch_size)
						y_pred = a.predict_batch(x_train)
						y_pred = y_pred.argmax(1)
						acc.update_state(y_train_1,y_pred)
						train_acc = acc.result().numpy()

						y_pred = a.predict_batch(x_test)
						y_pred = y_pred.argmax(1)
						acc.update_state(y_test_1,y_pred)
						test_acc = acc.result().numpy()
						t2 = time.time() - t1
						# print(layers,fids,lr,batch_size,epochs ,'train: ',train_acc,'\ttest :',test_acc, 'time:',t2)
						# writeFile.write(strng)					
						strng = str(layers)+';'+str(fids)+';'+str(lr)+';'+str(lamb)+';'+str(batch_size) +';'+str(epochs) +';train; '+str(train_acc) +';test ;'+str(test_acc) + ';time;'+str(t2) + '\n'
						print(strng)
						writeFile.write(strng)

# y_test_1 = y_test.argmax(1)
# y_train_1 = y_train.argmax(1)


# layers = (784,25,10)
# fids = (2,2)
# # fids = (4,4)
# lr = 0.5e-1
# batch_size = 10
# epochs = 10

# a = neuralNetwork(layers,fids)
# a.train(x_train,y_train,epochs,lr,1e-3,epochs)


# y_pred = a.predict_batch(x_train)
# y_pred = y_pred.argmax(1)
# acc.update_state(y_train_1,y_pred)
# train_acc = acc.result().numpy()

# y_pred = a.predict_batch(x_test)
# y_pred = y_pred.argmax(1)
# acc.update_state(y_test_1,y_pred)
# test_acc = acc.result().numpy()

# print('train:',train_acc,'\ttest:',test_acc)
#--------------------------------------------------------

# t2 = time.time() - t1

# y_pred = a.predict_batch(x_test)
# print(y_pred)
# y_pred = y_pred.argmax(1)
# y_train = y_train.argmax(1)
# y_test = y_test.argmax(1)
# # print(y_train.shape,y_pred.shape)
# # print([(np.argmax(y_train[i,:]),np.argmax(y_pred[i,:])) for i in range(y_pred.shape[0])])
# # print(y_pred)
# # print(y_pred)


# acc = metrics.Accuracy()
# acc.update_state(y_train,y_pred)
# acc.update_state(y_test,y_pred)
# print(acc.result().numpy())

end = time.time() -start
# print(layers,fids,lr,batch_size,epochs ,'train: ',train_acc,'\ttest :',test_acc, 'time:',end)
# writeFile.write('Total file run for: %d'%(end-start))


# writeFile.close()
print('time taken: ',end)
import time
start = time.time()

import csv
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics
import matplotlib.pyplot as plt

def get_data(filename):# gets the value and labels from the csv file 
	with open(filename) as f:
		reader = csv.reader(f,delimiter=',')
		file = np.array(list(reader)).astype('float')
		# file = np.array(list(reader)).astype('int')
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

def act_funcn_update(ip, fid):
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

def d_act_funcn(ip, fid):
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

def act_funcn(ip, fid):
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
	def __init__(self, layer_sizes, f_ids):
		self.layer_sizes = layer_sizes
		self.num_layers = len(layer_sizes)
		weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
		self.weights =  [0.1*np.random.standard_normal(s) for s in weight_shapes]
		self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]
		self.act_funcn_list = f_ids

	def predict(self, a): # for predicting a single data point
		for w,b,f in zip(self.weights,self.biases,self.act_funcn_list):
			a = act_funcn(np.dot(w,a) + b,f)
		return a
	def predict_batch(self, a):
		a = a.T
		for w,b,f in zip(self.weights,self.biases,self.act_funcn_list):
			a = act_funcn(np.dot(w,a) + b,f)
		return a.T

	def cost(self, pred, target): # add this
		''' Cost function for the model''' 
		pass

	def d_cost(self, pred, target): 
		# add regularisation terms
		# return softmax(pred) - target
		temp=0
		for w in self.weights:
			# print(temp)
			temp += np.linalg.norm(w) 
		return pred - target + self.lamb*temp

	def backprop1(self, inp, targets): #check
		''' Originally for multiple samples at once, but corrently incomplete''' 
		''' Works correctly for single data point inpiut'''
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

		err = self.d_cost(layer_ip[-1] , targets)* d_layer_ip[-1] 
		dw = []
		db = []
		db.append(err)
		dw.append(np.dot(err,layer_ip[-2].T))
		for i in range(2,(self.num_layers)):
			err = np.dot(self.weights[-i+1].T,err)*d_layer_ip[-i]
			db.insert(0,err)
			dw.insert(0,np.dot(err,layer_ip[-i-1].T))
		return dw,db
		

	def update_batch(self, x_batch, y_batch, lr):
	    """ lr- learning rate."""
	    l = x_batch.shape[0]
	    nabla_b = [np.zeros(b.shape) for b in self.biases]
	    nabla_w = [np.zeros(w.shape) for w in self.weights]
	    for x, y in zip(x_batch,y_batch):
	        delta_nabla_w, delta_nabla_b = self.backprop1(x, y)
	        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
	        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
	    self.weights = [w-(lr/l)*nw 
	                    for w, nw in zip(self.weights, nabla_w)]
	    self.biases = [b-(lr/l)*nb 
	                   for b, nb in zip(self.biases, nabla_b)]
	    return

	def train(self, xtrain, ytrain, epochs, lr, c, batch_size):
		self.epochs=epochs
		self.lr=lr
		self.lamb=c
		self.batch_size=batch_size
		for e in range(epochs):
			i =0
			while i < len(ytrain):
				xbatch = xtrain[i:i+batch_size,:]
				ybatch = ytrain[i:i+batch_size,:]
				# N = xbatch.shape[0]
				i += batch_size
				self.update_batch(xbatch,ybatch,lr)
		return

	def train_with_graph(self, x_train, y_train,x_test ,y_test, epochs, lr, c, batch_size):
		# fig = plt.figure()
		self.epochs=epochs
		self.lr=lr
		self.lamb=c
		self.batch_size=batch_size
		fig, (ax_tr,ax_ts) = plt.subplots(2,1)
		# ax_tr,ax_ts = fig.add_subplot(2,1)
		# ax = fig.add_subplot(iterations)
		# plt.title('Raw Feature Vectors',fontsize=16)
		txt = 'Layer size: '+str(self.layer_sizes)+"; fids: "+ str(self.act_funcn_list) + '; epochs: '+ str(epochs)+ '; lr: '+str(lr) + "; batch_size: "+ str(batch_size)
		plt.figtext(0,1,txt)
		plt.title('Using 25 PCA features',fontsize=16)
		acc = metrics.Accuracy()
		y_test_1 = y_test.argmax(1)
		y_train_1 = y_train.argmax(1)
		train_accuracy = []
		test_accuracy = []
		idx = 0
		for e in range(epochs):
			i =0
			while i < len(y_train):
				x_batch = x_train[i:i+batch_size,:]
				y_batch = y_train[i:i+batch_size,:]
				# N = xbatch.shape[0]
				i += batch_size
				self.update_batch(x_batch,y_batch,lr)
				#train accuracy
				y_pred = a.predict_batch(x_train)
				y_pred = y_pred.argmax(1)
				acc.update_state(y_train_1,y_pred)
				train_accuracy.append(acc.result().numpy())
				#test accuracy
				y_pred = a.predict_batch(x_test)
				y_pred = y_pred.argmax(1)
				acc.update_state(y_test_1,y_pred)
				# test_acc = acc.result().numpy()
				test_accuracy.append(acc.result().numpy())
				idx += 1
		xrang = np.arange(idx)
		ax_ts.plot(xrang,test_accuracy)
		ax_ts.set_xticks([])
		ax_ts.set_title('Test Accuracy')
		ax_tr.set_xticks([])
		ax_tr.plot(xrang,train_accuracy)
		ax_tr.set_title('Train Accuracy')
		t = time.time() - start
		txt = 'final: test accuracy = '+str(test_accuracy[-1])+ '; train_accuracy = '+ str(train_accuracy[-1])
		plt.figtext(0,0,txt)
		fig.savefig('fig_%d.png'%t,bbox_inches='tight')

		plt.close()
		return


filename = 'data.csv'
# filename = 'pca_data.csv'
data,labels = get_data(filename)

data = data/255.0
targets = to_categorical(labels)

x_train, x_test,y_train,y_test = train_test_split(data,targets, train_size=0.25,test_size=0.25,random_state=13)
y_test_1 = y_test.argmax(1)
y_train_1 = y_train.argmax(1)

layers = (784,10)
fids = (4,0)

# layers = (784,25,10)
# layers = (25,10)
# layers = (25,16,10)
# fids = (2,2)
# train: 0.9395555        test: 0.92433333
# time taken:  7.893457412719727


# fids = (4,4)
# train: 0.896    test: 0.8886667
# time taken:  7.4340431690216064

# fids = (1,1)
# train: 0.5262222        test: 0.5226667
# time taken:  8.432704448699951

# layers = (784,400,25,10) 
# fids = (2,2,2)
# train: 0.988    test: 0.9676667
# time taken:  105.92944073677063

# lr = 0.5e-1
lr = 1e-1
batch_size = 10
epochs = 50
lamb = 1e-3 
a = neuralNetwork(layers,fids)
a.train(x_train,y_train,epochs,lr,lamb,epochs)
# a.train_with_graph(x_train,y_train,x_test,y_test,epochs,lr,lamb,batch_size)


acc = metrics.Accuracy()

y_pred = a.predict_batch(x_train)
y_pred = y_pred.argmax(1)
acc.update_state(y_train_1,y_pred)
train_acc = acc.result().numpy()

y_pred = a.predict_batch(x_test)
y_pred = y_pred.argmax(1)
acc.update_state(y_test_1,y_pred)
test_acc = acc.result().numpy()

print('train:',train_acc,'\ttest:',test_acc)

end = time.time() - start
print('time taken: ',end)
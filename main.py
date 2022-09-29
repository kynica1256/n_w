import numpy as np

np.random.seed(1)

#in_data_1 = np.random.random((1,10))

#in_data_2 = np.random.random((1,10))


def sigmoid(x):
	return 1/(1+np.exp(-x))

def relu(x):
	return (x>0)*x

def relu2deriv(output):
	return output>0

train_data = np.array([
	[0,1],
	[1,0]
	])

correct_data = np.array([[1,0]]).T

w1 = 2*np.random.random((2, 500))-1
w2 = 2*np.random.random((500, 500))-1
w3 = 2*np.random.random((500, 1))-1

alpha = 0.2

for i in range(50):
	error = 0
	for b in range(len(train_data)):
		layer0 = train_data[b]
		print(layer0)
		layer1 = relu(np.dot(layer0, w1))
		layer2 = relu(np.dot(layer1, w2))
		layer3 = sigmoid(np.dot(layer2, w3))

		error += np.sum((layer3-correct_data[b])**2)

		delta3 = layer3-correct_data[b]
		print(layer2.T)
		delta2 = delta3.dot(w3.T) * relu2deriv(layer2)
		delta1 = delta2.dot(w2.T) * relu2deriv(layer1)
		print(delta3)
		w3 -= alpha * np.array([layer2]).T.dot(np.array([delta3]))
		w2 -= alpha * np.array([layer1]).T.dot(np.array([delta2]))
		w1 -= alpha * np.array([layer0]).T.dot(np.array([delta1]))
	print(error)

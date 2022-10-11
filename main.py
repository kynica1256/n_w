import numpy as np
import pandas as pd
import json
import sys
from data_sort import data_sort
from tqdm import tqdm
import time


np.random.seed(1)

with open("ds.json", "r", encoding="utf-8") as f:
	ds = json.loads(f.read())
	f.close()

df = pd.read_csv('IMDB Dataset.csv/IMDB Dataset.csv')


d_s = data_sort(ds["data"], df["review"], df["sentiment"])

train_data, correct_data_train = d_s.method_sort_data(2000)

test_data, correct_data_test = d_s.method_sort_data(800)


def sigmoid(x):
	return 1/(1+np.exp(-x))

def relu(x):
	return (x>0)*x

def relu2deriv(output):
	return output>0


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()





w1 = 0.2*np.random.random((200, 200))-0.1
w2 = 0.2*np.random.random((200, 125))-0.1
w3 = 0.2*np.random.random((125, 1))-0.1



def neural_network_mini(data, N=0):
	if N != 0:
		layer0 = data[N]
	else:
		layer0 = data
	layer1 = relu(np.dot(layer0, w1))
	dropout_mask = np.random.randint(2, size=layer1.shape)
	layer1*=dropout_mask*2
	layer2 = relu(np.dot(layer1, w2))
	layer3 = sigmoid(np.dot(layer2, w3))
	return layer3



alpha = 0.005

m_o = 0
m_o_ = len(train_data) // len(test_data)

 
for i in range(len(train_data)*3):
	error_test = 0
	error = 0
	correct_cnt = 0
	pbar = tqdm(total=100, ncols=100)
	for b in np.random.randint(1, len(train_data)-1, 100).tolist():
		#liner_data = "\t[{n}]".format(n="".join([" " for i in range(20)]))
		for c in range(20):
			layer0 = train_data[b]
			layer1 = relu(np.dot(layer0, w1))
			dropout_mask = np.random.randint(2, size=layer1.shape)
			layer1*=dropout_mask*2
			layer2 = relu(np.dot(layer1, w2))
			dropout_mask_1 = np.random.randint(2, size=layer2.shape)
			layer2*=dropout_mask_1
			layer3 = np.dot(layer2, w3)
			#if np.sum(layer3) > 1 or np.sum(layer3) < 0:
			#	continue
			layer3 = sigmoid(layer3)
			#print(layer3)
			#print(correct_data_train[b].T)
			error += np.sum((layer3-correct_data_train[b].T)**2)
			correct_cnt += int(np.argmax(layer3) == np.argmax(correct_data_train[b].T))
			delta3 = layer3 - correct_data_train[b].T
			delta3 = layer3-correct_data_train[b].T
			delta2 = delta3.dot(w3.T) * relu2deriv(layer2)
			delta2*=dropout_mask_1
			delta1 = delta2.dot(w2.T) * relu2deriv(layer1)
			delta1*=dropout_mask
			w3 -= alpha * np.array([layer2]).T.dot(np.array([delta3]))
			w2 -= alpha * np.array([layer1]).T.dot(np.array([delta2]))
			w1 -= alpha * np.array([layer0]).T.dot(np.array([delta1]))
		#time.sleep(0.5)
		pbar.update(1)
	pbar.close()
	print("\n")
	#print(error, end=" error train\n")
	#print("\n\n")
	m_o += 1
	if m_o == m_o_:
		#print("\n")
		rand_ = np.random.randint(1, len(test_data)-1)
		m_o = 0
		res = neural_network_mini(test_data, rand_)
		#error_test += np.sum((layer3-correct_data_test[rand_].T)**2)
		error_test += np.sum((res-correct_data_test[rand_].T)**2)
		print("---------")
		print(error, end="\t error train\n")
		print(error_test, end="\t error test\n")
		print("---------")
		print("\n")


data_i = d_s.practice_data(input())

res = neural_network_mini(data_1)

data_mark = ["negative", "positive"]

print(res)

print(data_mark[round(np.sum(res))])

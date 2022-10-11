import numpy as np

class data_sort(object):
	data_id_word_ = dict()
	def __init__(self, data_json_lang, data_train_test, correct_data_):
		self.data_json_lang = data_json_lang
		self.data_train_test = data_train_test
		self.correct_data_ = correct_data_

	def method_sort_data(self, num):
		#self.data_id_word_ = dict()
		correct_data = np.array([])
		data_mark = {
			"negative":1,
			"positive":0
		}
		for i in self.data_json_lang:
			self.data_id_word_[i] = float('{:.3f}'.format(np.random.random_sample()))

		data_ = np.zeros((1,200))
		for i in np.random.randint(0, 8000, num).tolist():
			i_ = self.data_train_test[i]
			if len(i_) > 700:
				continue
			correct_data = np.append(correct_data, data_mark[self.correct_data_[i]])
			text_ = i_.lower().split(" ")
			res_text = np.zeros((1,len(text_)))
			for b in range(len(text_)):
				data_txt = self.data_json_lang[b]
				if text_[b] in self.data_json_lang:
					res_text[0][b] = self.data_id_word_[data_txt]
			res_text = np.array(list(filter((0.0).__ne__, res_text[0])))
			if len(res_text) < 200:
				for i in range(200-len(res_text)):
					res_text = np.append(res_text, 0.0)
			data_ = np.vstack([data_, res_text])
		return (data_, correct_data)
	def practice_data(self, text):
		text_ = i.lower().split(" ")
		res_text = np.zeros((1,len(text_)))
		for b in range(len(text_)):
			data_txt = self.data_json_lang[b]
			if text_[b] in self.data_json_lang:
				res_text[0][b] = self.data_id_word_[data_txt]
		res_text = np.array(list(filter((0.0).__ne__, res_text[0])))
		if len(res_text) < 200:
			for i in range(200-len(res_text)):
				res_text = np.append(res_text, 0.0)
		data_ = np.vstack([data_, res_text])
		return data_
		

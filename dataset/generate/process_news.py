import numpy as np
data_ = np.loadtxt('dataset/generate/news_p.csv', delimiter=',')

doc_id_map = {}
word_id_map = {}
doc_id_list = list(set(data_[:,0]))
word_id_list = list(set(data_[:,1]))

count = 0
for doc in doc_id_list:
    doc_id_map[doc] = count
    count+=1

count = 0
for word in word_id_list:
    word_id_map[word] = count
    count +=1

data_use = np.zeros([len(doc_id_list), len(word_id_list)])
for _ in range(data_.shape[0]):
    doc_id = doc_id_map[data_[_, 0]]
    word_id = word_id_map[data_[_, 1]]
    word_freq = data_[_, 2]
    data_use[doc_id, word_id] = word_freq


np.save('news_pp', data_use)
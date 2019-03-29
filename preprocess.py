import csv
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from utils import save_json

# raw data
glove = '../datasets/glove/glove.840B.300d.txt'
data = '../datasets/quora_insincere_questions/train.csv'

# where to store processed data
X_train_path = 'data/train/X_train.npy'
Y_train_path = 'data/train/Y_train.npy'
X_test_path = 'data/test/X_test.npy'
Y_test_path = 'data/test/Y_test.npy'
embeddings_path = 'data/embeddings/embeddings.npy'
info_path = 'data/info.json'

assert os.path.exists(glove)
assert os.path.exists(data)

# read data
X = []
Y = []

with open(data, encoding = 'utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in tqdm(reader):
        X.append(row['question_text'])
        Y.append(1 if row['target'] == '1' else 0)
Y = np.array(Y)

# Tokenize texts 
t = Tokenizer()
t.fit_on_texts(X)
X = t.texts_to_sequences(X)
word_index = t.word_index
vocab_size = len(word_index) + 1


# pad sequences
max_len = 50
X = pad_sequences(X, maxlen=max_len, padding="post")

# split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)
print("X_train shape : {}, X_test shape : {}".format(X_train.shape, X_test.shape))

# Create embedding dict
embed_dict = {}
with open(glove, encoding='utf-8') as f:
    for line in tqdm(f):
        values = line.split(' ')
        embed_dict[values[0]] = np.asarray(values[1:], dtype='float32')
        
#Create word embeddings
embed_dim = 300
embeddings = np.zeros((vocab_size, embed_dim))
for word, i in tqdm(word_index.items()):
    e = embed_dict.get(word)
    if e is not None:
        embeddings[i] = e
    else:
        embeddings[i] = np.random.uniform(-0.25, 0.25, embed_dim)

# save data
np.save(open(X_train_path, 'wb+'), X_train)
np.save(open(X_test_path, 'wb+'), X_test)
np.save(open(Y_train_path, 'wb+'), Y_train)
np.save(open(Y_test_path, 'wb+'), Y_test)
np.save(open(embeddings_path, 'wb+'), embeddings)

info = {
	'embed_dim' : embed_dim,
	'max_len' : max_len,
	'vocab_size' : vocab_size
}
save_json(info, info_path)


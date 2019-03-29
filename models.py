from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization

from utils import *

def create_embeddings(config):
	embeddings = load_numpy('data/embeddings/embeddings.npy')
	e = Embedding(config['vocab_size'], config['embed_dim'], input_length=config['max_len'], weights=[embeddings], mask_zero=True, trainable=False)
	return e
def bilstm(config):

	sentence_input = Input(shape=(config['max_len'],))

	embeddings = create_embeddings(config) (sentence_input)
	lstm = Bidirectional(LSTM(config['embed_dim'])) (embeddings)
	dense = Dense(config['hidden_dim'], activation='relu', kernel_regularizer=regularizers.l2(config['l2']))(lstm)
	dense = Dropout(config['dropout'])(dense)
	dense = BatchNormalization() (dense)
	dense = Dense(config['hidden_dim'], activation='relu', kernel_regularizer=regularizers.l2(config['l2']))(lstm)
	dense = Dropout(config['dropout'])(dense)
	dense = BatchNormalization() (dense)
	dense = Dense(1, activation='sigmoid') (dense)
	model = Model(inputs = [sentence_input], outputs=dense)

	optimizer = Adam(lr = config['lr'])
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	model.summary()
	return model





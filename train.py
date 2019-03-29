import argparse
import os

from utils import *
import models
import importlib
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

dataset_params = 'data/info.json'

def train_and_evaluate(model, config, data):
	checkpoint = ModelCheckpoint(filepath=os.path.join(config['training_dir'], "{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.hdf5"))
	reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
	early_stop = EarlyStopping(patience=1)

	# resume training
	if config['restore'] is not None:
		model.load_weights(config['weights'])

	print("Starting training...")
	history = model.fit(data['X_train'], data['Y_train'], epochs = config['epochs'], initial_epoch=config['initial_epoch'], verbose=1, batch_size=config['batch_size'], callbacks = [checkpoint, reducelr, early_stop], validation_split=0.1)

	print("Finished training")
	
	# save model weights?
	if config['save']:
		model.save_weights(os.path.join(config['training_dir'], '{}.hdf5'.format(config['model'])))

	# plot history
	history_dict = history.history
	acc = history_dict['acc']
	val_acc = history_dict['val_acc']
	loss = history_dict['val_loss']
	val_loss = history_dict['val_loss']

	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, loss, 'bo', label='loss')
	plt.plot(epochs, val_loss, 'b', label='val loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(os.path.join(config['training_dir'], 'loss.png'))


	plt.clf()
	plt.plot(epochs, acc, 'bo', label='acc')
	plt.plot(epochs, val_acc, 'b', label='val acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(os.path.join(config['training_dir'], 'accuracy.png'))

	# evaluate and save results
	loss, accuracy = model.evaluate(data['X_test'], data['Y_test'])
	print('loss: {.2f}, accuracy :{.2f}'.format(loss, accuracy))
	save_json({'acc': accuracy, 'loss' : loss}, os.path.join(config['training_dir'], 'evaluation.json'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='name of model to train', default='bilstm')
	parser.add_argument('--training_dir', help='directory in experiments/ containing training_parameters.json file', default = 'bilstm')
	parser.add_argument('--restore', help='.hdf5 file in model_dir to load weights from')
	parser.add_argument('--save', help='save model weights after training', dest='save', action='store_true')

	args = parser.parse_args()
	config = {}
	training_dir = os.path.join('experiments', args.training_dir)
	assert os.path.exists(training_dir)
	config['training_dir'] = training_dir
	config['model'] = args.model

	# resume training
	if args.restore is not None:
		config['restore'] = os.path.join(training_dir, args.restore)
		assert os.path.exists(config['restore'])
		config['initial_epoch'] = get_initial_epoch(config['restore']) #point where to start training
	else:
		config['restore'] = None
		config['initial_epoch'] = 0

	# save model weights after training
	config['save'] = args.save
	# training parameters
	training_parameters =  os.path.join(training_dir, 'training_parameters.json')
	assert os.path.exists(training_parameters)
	config.update(read_json(dataset_params, training_parameters))

	# create model
	function = getattr(models, args.model)
	model = function(config)
	data = load_data()
	train_and_evaluate(model, config, data)
	# print (module())
	




	




import json
import os
import numpy as np
def save_json(_dict, dest):
	with open(dest, 'w') as f:
		d = {k : v for k, v in _dict.items()}
		json.dump(d, f, indent=4)
def read_json(*files):
	config = {}
	for file in files:
		with open(file, "r") as f:
			data = json.load(f)
			for k, v in data.items():
				config[k] = v
	return config
def get_basename(filename):
	return os.path.splitext(os.path.basename(filename))[0]

def get_initial_epoch(checkpoint_file):
	base = get_basename(checkpoint_file)
	return int(base.split("-")[0])

def load_numpy(filename):
	return np.load(open(filename, 'rb'))

def load_data():
	# return dictionary object where the keys are X_train, Y_train, X_test, Y_test, embeddings and the values are the corresponding numpy array objects
	files = [os.path.join(root, f) for root, d, filenames in os.walk('data') for f in filenames if os.path.splitext(f)[1] == '.npy']
	data = {get_basename(i) : i for i in files}
	data = {name : load_numpy(file) for name, file in data.items()}
	return data

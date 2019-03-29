import os
from urllib.request import urlopen
import requests
from zipfile import ZipFile
from io import BytesIO

glove_url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
dest = "../datasets/glove/"

# download glove
os.makedirs(dest, exist_ok=True)
response = urlopen(glove_url)
zipfile = ZipFile(BytesIO(response.read()))
zipfile.extractall(dest)




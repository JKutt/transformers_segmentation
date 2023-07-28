import numpy as np
import matplotlib
from pandas import DataFrame
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from random import seed
import random
import time
import gzip
from urllib.request import urlopen
import pandas as pd
import io
import requests

# filter for types of structures you want in the model
his_filter=['FOLD','FAULT','DYKE', 'PLUG', 'UNCONFORMITY']

# how many model do you want?
num_models = 10

# read the model list
models=pd.read_csv('model_list/models.csv')
models2=models[models['event03'].str.contains(his_filter[0]) & models['event04'].str.contains(his_filter[1]) & models['event05'].str.contains(his_filter[2])] 
models2=models2.reset_index(drop=True)
model_number2=len(models2)

if(len(models2)):
    print("sampling from",len(models2),"models matching filter",his_filter)
else:
    print("no models found with filter", his_filter, "check list syntax and spelling of events")

#seed random number generator
now = time.time()
seed(int(now))

url='https://cloudstor.aarnet.edu.au/plus/s/8ZT6tjOvoLWmLPx/download?path=%2f'
used=[]
z=0
fail=0

for ii in range(num_models):

    used = []
    ran =random.randint(0,model_number2-1) 
    if(ran in used):
        pass
    else:
        used.append(ran)
    file_split=models2.iloc[ran]['root'].split('/')  
    tail=models2.iloc[ran]['event03']+'_'+models2.iloc[ran]['event04']+'_'+models2.iloc[ran]['event05']+'&files='
    root=url+tail+file_split[2]
    path=root+'.g12.gz'
    path_sus=root+'.g00.gz'

    print(f"grabbing file: {path}")
    my_gzip_stream = urlopen(path)
    my_stream = gzip.open(my_gzip_stream, 'r')

    models = np.loadtxt(my_stream,skiprows=0)

    print(f"grabbing file: {path_sus}")
    my_gzip_stream = urlopen(path_sus)
    my_stream = gzip.open(my_gzip_stream, 'r')

    properties_from_file = []
    for line in my_stream:

        if 'Sus' in line.decode("utf-8"):

            properties_from_file.append(line.decode("utf-8").split()[-1])

    properties_from_file = list(map(float, properties_from_file))

    mod2=models.reshape((200,200,200))
    mod2=np.transpose(mod2,(0,2,1))
    mod2.shape

    # -------------------------------------------------------------------------------------------

    # assign physical properties

    #

    # generate the physical properties
    units_common = list(set(mod2.flatten()))
    print(units_common)
    print(properties_from_file)

    num_units = len(properties_from_file)

    sus_model = np.zeros_like(mod2)

    for unit in units_common:

        index_unit = mod2 == unit
        print(unit, int(unit)-1, len(properties_from_file))
        sus_model[index_unit] = properties_from_file[int(unit) - 1]

    # ----------------------------------------------------------------------------------------------

    # save data

    np.save(f"C:/Users/johnk/Documents/git/DeepMagnetics/models/model{ii}.npy", sus_model)


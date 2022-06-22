from pearson_hash import *
import numpy as np
import os
import pickle

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)        

while True:
    files = os.listdir("./weights")
    hashDict = {}
    for file in files:
        if ".dat" in file:
            index = file.find(".")
            layer = file[:index]
            hashDict[layer] = main(layer)
            
    save_dict(hashDict, "fpgahashes.pkl")
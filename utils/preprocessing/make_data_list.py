import glob
import random

def make_data_list(path, ftype="flac", shuffle=True):
    dataset = []
    for path in glob.iglob(path+f'*/*.{ftype}'):
        data_pair = []
        data_pair.append(path)
        dataset += [data_pair]
    if(shuffle) :
        random.seed(1024)
        random.shuffle(dataset)
    return dataset

def make_trainlist(path, ftype="flac", shuffle=True):
    dataset = []
    for path in glob.iglob(path+f'*/*.{ftype}'):
        data_pair = []
        data_pair.append(path)
        label = path.split('/')
        data_pair.append(label[-2])
        dataset += [data_pair]
    if(shuffle) :
        random.seed(1024)
        random.shuffle(dataset)
    return dataset
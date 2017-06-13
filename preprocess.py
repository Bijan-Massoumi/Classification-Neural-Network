import numpy as np
import csv

def load_images(filename):
    with open(filename) as f:
        f = list(csv.reader(f))[1:]
        y = [None]*len(f)
        images = [None]*len(f)
        for i,row in enumerate(f):
            images[i] = np.array(f[i][1:],dtype = 'float64')
            y[i] = np.array(get_correct_target(int(f[i][0])),dtype = 'float64')
        return np.array(images), np.array(y)

def partition_set(portion,images,targets):
    dividing_index = int(portion * targets.shape[0])
    image_train = images[:dividing_index]
    target_train = targets[:dividing_index]

    image_test = images[dividing_index:]
    target_test = targets[dividing_index:]

    return image_train,target_train,image_test,target_test





def get_correct_target(num):
    return {
        0:[1,0,0,0,0,0,0,0,0,0],
        1:[0,1,0,0,0,0,0,0,0,0],
        2:[0,0,1,0,0,0,0,0,0,0],
        3:[0,0,0,1,0,0,0,0,0,0],
        4:[0,0,0,0,1,0,0,0,0,0],
        5:[0,0,0,0,0,1,0,0,0,0],
        6:[0,0,0,0,0,0,1,0,0,0],
        7:[0,0,0,0,0,0,0,1,0,0],
        8:[0,0,0,0,0,0,0,0,1,0],
        9:[0,0,0,0,0,0,0,0,0,1]
    }[num]

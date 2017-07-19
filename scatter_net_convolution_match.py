from scatter_net_convolution_train import forwardprop
from scatter_net_convolution_train import init_weights
from scatter_net_convolution_train import init_bias
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import time
import argparse, os
import random

def get_spect(data,singular = False):
    y_file = data+"_val.csv"
    x_file = data+".csv"

    X = np.transpose(np.genfromtxt(x_file,delimiter=','))
    Y = np.genfromtxt(y_file,delimiter=',')

    x = (list(X.shape))
    x.append(1)
    X = np.reshape(X,x)
    if singular == False:
    	index = random.choice(list(range(len(Y))))
    	return np.array([X[index]]), np.array([Y[index]])
    else:
        return np.array([X]), np.array([Y])

def main(data,reuse_weights,output_folder,weight_name_save,weight_name_load,n_batch,numEpochs,lr_rate,lr_decay,num_layers,n_hidden,percent_val,kernel_size,kernel_no):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    test_X, test_Y  = get_spect(data,singular=True)
    
    x_size = test_X.shape[1]
    y_size = test_Y.shape[1]

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size, 1])
    y = tf.placeholder("float", shape=[None, y_size])
    weights = []
    biases = []
    # Weight initializations
    for i in range(0,num_layers):
        if i == 0:
            weights.append(init_weights((kernel_size,1,kernel_no)))
            biases.append(init_bias(kernel_no))
        elif i==1:
            weights.append(init_weights((int(0.5*(x_size-kernel_size+1))*kernel_no,n_hidden)))
            biases.append(init_bias(n_hidden))
        else:
            weights.append(init_weights((n_hidden,n_hidden)))
            biases.append(init_bias(n_hidden))
    weights.append(init_weights((n_hidden,y_size)))
    biases.append(init_bias(y_size))
    # Forward propagation
    yhat    = forwardprop(X, weights,biases,num_layers)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,output_folder+weight_name_save+".ckpt")
        out = sess.run(yhat,feed_dict = {X:test_X,y:test_Y})
        print("Computed: "+str(out))
        print("Expected: "+str(test_Y))
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data",type=str,default='data/test')
    #parser.add_argument("--data",type=str,default='data/CompleteDataFiles/3_layer_tio2_fixed_06_21_1')
    parser.add_argument("--reuse_weights",type=str,default='False')
    parser.add_argument("--output_folder",type=str,default='results/3_Layer_TiO2_20Kernel_Convolution_5layers_650per_Positive/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--weight_name_save",type=str,default="Weights_and_Biases")
    parser.add_argument("--n_batch",type=int,default=100)
    parser.add_argument("--numEpochs",type=int,default=100)
    parser.add_argument("--lr_rate",default=0.000001)
    parser.add_argument("--lr_decay",default=.9)
    parser.add_argument("--num_layers",default=5)
    parser.add_argument("--n_hidden",default=650)
    parser.add_argument("--percent_val",default=.2)
    parser.add_argument("--kernel_size",default=5)
    parser.add_argument("--kernel_no",default=20)

    args = parser.parse_args()
    dict = vars(args)

    for i in dict:
        if (dict[i]=="False"):
            dict[i] = False
        elif dict[i]=="True":
            dict[i] = True

    kwargs = {
            'data':dict['data'],
            'reuse_weights':dict['reuse_weights'],
            'output_folder':dict['output_folder'],
            'weight_name_save':dict['weight_name_save'],
            'weight_name_load':dict['weight_name_load'],
            'n_batch':dict['n_batch'],
            'numEpochs':dict['numEpochs'],
            'lr_rate':dict['lr_rate'],
            'lr_decay':dict['lr_decay'],
            'num_layers':dict['num_layers'],
            'n_hidden':dict['n_hidden'],
            'percent_val':dict['percent_val'],
            'kernel_size':dict['kernel_size'],
            'kernel_no':dict['kernel_no']}

    main(**kwargs)

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import time
import argparse, os

seed = time.time()
tf.set_random_seed(seed)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=.001)
    return tf.Variable(weights)

def init_bias(shape):
    """ Weight initialization """
    biases = tf.random_normal([shape], stddev=.001)
    return tf.Variable(biases)

def get_data(data,percentTest=0.2,random_state=42):
    x_file = data+".txt"
    y_file = data+".csv"
    file = open(x_file)

    train_X = []
    for line in file:
        x = []
        for char in line[:25]:
            x.append(int(char))
        train_X.append(x)
    train_X = np.array(train_X)
    train_Y = np.genfromtxt(y_file,delimiter=',')
    print(train_X.shape,train_Y.shape)
    X_train, X_val, y_train, y_val = train_test_split(train_X,train_Y,test_size=percentTest,random_state=random_state)
    return X_train, y_train, X_val, y_val

def forwardprop(X, weights, biases, num_layers,dropout=False):
    htemp = None
    for i in range(0, num_layers):
        if i ==0:
            htemp = tf.add(tf.nn.relu(tf.matmul(X,weights[i])),biases[i])    
        else:   
            htemp = tf.add(tf.nn.relu(tf.matmul(htemp,weights[i])),biases[i])
        print("Bias: " , i, " : ", biases[i])
    yval = tf.add(tf.matmul(htemp,weights[-1]),biases[-1])
    return yval


def main(data,reuse_weights,output_folder,weight_name_save,weight_name_load,n_batch,numEpochs,lr_rate,lr_decay,num_layers,n_hidden,percent_val):
    lr_rate = float(lr_rate)
    lr_decay = float(lr_decay)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_X, train_Y , val_X, val_Y = get_data(data,percentTest=percent_val)

    x_size = train_X.shape[1]
    y_size = train_Y.shape[1]

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    weights = []
    biases = []
    # Weight initializations
    for i in range(0,num_layers):
        if i ==0:
            weights.append(init_weights((x_size,n_hidden)))
        else:
            weights.append(init_weights((n_hidden,n_hidden)))
        biases.append(init_bias(n_hidden))
    weights.append(init_weights((n_hidden,y_size)))
    biases.append(init_bias(y_size))
    # Forward propagation
    yhat    = forwardprop(X, weights,biases,num_layers)
   
    # Backward propagation
    cost = tf.reduce_sum(tf.square(y-yhat))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate, decay=lr_decay).minimize(cost)
    #MAY HAVE TO CHANGE THESE TWO INTO CROSS ENTROPY AND SOME OTHER OPTIMIZER

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if reuse_weights:
            saver.restore(sess,output_folder+weight_name_save+".ckpt")
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
        step = 0
        curEpoch=0
        cum_loss = 0
        numFile = 0 
        while True:
            train_file_name = output_folder+"train_train_loss_" + str(numFile) + ".txt"
            if os.path.isfile(train_file_name):
                numFile += 1
            else:
                break
        train_loss_file = open(train_file_name,'w')
        val_loss_file = open(output_folder+"train_val_loss_"+str(numFile) + "_val.txt",'w')
        start_time=time.time()
        print("========                         Iterations started                  ========")
        while curEpoch < numEpochs:
            batch_x = train_X[step * n_batch : (step+1) * n_batch]
            batch_y = train_Y[step * n_batch : (step+1) * n_batch]
            val_loss = sess.run(cost,feed_dict={X:val_X,y:val_Y})
            print(val_loss)
            sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
            cum_loss += sess.run(cost,feed_dict={X:batch_x,y:batch_y})
            step += 1
            if step == int(train_X.shape[0]/n_batch): #Epoch finished
                step = 0
                curEpoch +=1          
                train_loss_file.write(str(float(cum_loss))+str("\n"))
                if (curEpoch % 10 == 0 or curEpoch == 1):
                    #Calculate the validation loss
                    val_loss = sess.run(cost,feed_dict={X:val_X,y:val_Y})
                    print("Validation loss: " , str(val_loss))
                    val_loss_file.write(str(float(val_loss))+str("\n"))
                    val_loss_file.flush()

                    print("Epoch: " + str(curEpoch+1) + " : Loss: " + str(cum_loss))
                    train_loss_file.flush()
                cum_loss = 0
        save_path = saver.save(sess,output_folder+weight_name_save+".ckpt")
        print("Saved to: "+save_path)
    print("========Iterations completed in : " + str(time.time()-start_time) + " ========")
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data",type=str,default='MetasurfaceData/spectra')
    parser.add_argument("--reuse_weights",type=str,default='False')
    parser.add_argument("--output_folder",type=str,default='MetasurfaceResults/4x200/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--weight_name_save",type=str,default="")
    parser.add_argument("--n_batch",type=int,default=100)
    parser.add_argument("--numEpochs",type=int,default=200)
    parser.add_argument("--lr_rate",default=0.00001)
    parser.add_argument("--lr_decay",default=.9)
    parser.add_argument("--num_layers",default=4)
    parser.add_argument("--n_hidden",default=200)
    parser.add_argument("--percent_val",default=.2)

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
            'percent_val':dict['percent_val']
            }

    main(**kwargs)

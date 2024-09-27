import speck as sp
import numpy as np
from time import time

from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

import os
import tensorflow as tf

# Set the number of threads used for parallelism between independent operations
tf.config.threading.set_inter_op_parallelism_threads(1)
# Set the number of threads used within an individual op for parallelism
tf.config.threading.set_intra_op_parallelism_threads(1)

#load distinguishers
json_file = open('../../single_block_resnet.json','r');
json_model = json_file.read();
net5 = model_from_json(json_model);
net6 = model_from_json(json_model);
net7 = model_from_json(json_model);
net8 = model_from_json(json_model);

net5.load_weights('../../net5_small.h5');
net6.load_weights('../../net6_small.h5');
net7.load_weights('../../net7_small.h5');
net8.load_weights('../../net8_small.h5');

def evaluate(net,X,Y):
    t0 = time();
    Z = net.predict(X,batch_size=1<<14,verbose=0).flatten();
    t1 = time();
    Zbin = (Z > 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);
    print("Wall time (in seconds): ", (t1 - t0), " = 2^", np.log2(t1 - t0), " seconds\n");

TDN = 1 << 19
X5,Y5 = sp.make_train_data(TDN, 5);
X6,Y6 = sp.make_train_data(TDN, 6);
X7,Y7 = sp.make_train_data(TDN, 7);
X8,Y8 = sp.make_train_data(TDN, 8);

print('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting');
print('5 rounds:');
evaluate(net5, X5, Y5);
print('6 rounds:');
evaluate(net6, X6, Y6);
print('7 rounds:');
evaluate(net7, X7, Y7);
print('8 rounds:');
evaluate(net8, X8, Y8);

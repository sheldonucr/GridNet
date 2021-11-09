#!/usr/bin/env python



#---------- Package Import ----------

import sys
import math
from time import time

import tensorflow as tf
from tensorflow.keras.layers import Concatenate
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import glob

from EDG_model import autoencoder

#---------- end - Package Import ----------



#---------- Setting ----------

setting = {
        'image_height': 32,
        'image_width': 32,
        'circut_height': 200000,
        'circut_width': 200000,
    }

#---------- end - Setting ----------



#---------- Function ----------

def read_file(path, setting):

    image_height = setting.get('image_height')
    image_width  = setting.get('image_width')
    circut_height = setting.get('circut_height')
    circut_width = setting.get('circut_width')

    path += ".sp"
    glob_result = glob.glob(path)
    num_train = len(glob.glob(path))

    power_map = np.zeros((num_train, image_height, image_width))
    density_map = np.zeros((num_train, image_height, image_width))
    distance_map = np.zeros((num_train, image_height, image_width))
    IRDrop_map = np.zeros((num_train, image_height, image_width))

    for im_num, fname in enumerate(glob_result):
        voltage = 1.1
        power_pad_position = []
        # Read from .sp file
        with open(fname) as file:
            lines = file.readlines()
            # Read each line
            for line in lines:
                if( line[0] in (".","\n") or len(line)==0 ): continue
                # Resistance
                if( line[0] == "R" ): continue
                # Current source
                elif( line[0] in "Ii" ):
                    items = line.split()
                    vdd = items[1].split('_')
                    x = int( int(vdd[1]) * image_height / circut_height )
                    y = int( int(vdd[2]) * image_width / circut_width )
                    power_map[im_num,x,y] += float(items[3])
                # Voltage source
                elif( line[0] in "Vv" ):
                    items = line.split()
                    vdd = items[1].split('_')
                    x = int( int(vdd[1]) * image_height / circut_height )
                    y = int( int(vdd[2]) * image_width / circut_width )
                    power_pad_position.append((x,y))
                    # Record VDD
                    voltage = float(items[3])
        # Read from .ic0 file
        IRDrop_temp = np.zeros((image_height, image_width))
        flag = ""
        with open(fname[:-3]+".ic0") as file:
            lines = file.readlines()
            # Read each line
            for line in lines:
                # Ignore comments
                if( line[0] in ("*","\n") ): continue
                # Update flag
                elif( line[0] == "." ):
                    flag = line[1:-1]
                    continue
                # Check flag
                if( flag != "nodeset" ): continue

                items = line.split()
                u = items[1]
                vdd = u.split('_')
                if( vdd[3] != "1" ): continue # Only read the M1 layer
                x = int( int(vdd[1]) * image_height / circut_height )
                y = int( int(vdd[2]) * image_width / circut_width )
                IRDrop_map[im_num,x,y] += voltage - float(items[3])
                IRDrop_temp[x,y] += 1
        # Compute mean IR Drop in each pixel
        for x in range(image_height):
            for y in range(image_width):
                if( IRDrop_temp[x,y] > 0 ):
                    IRDrop_map[im_num,x,y] /= IRDrop_temp[x,y]
        # Compute effective distance
        if( len(power_pad_position) > 0 ):
            for x in range(image_height):
                for y in range(image_width):
                    for position in power_pad_position:
                        if( x == position[0] and y == position[1] ):
                            distance_map[im_num,x,y] = 0
                            break
                        else:
                            distance_map[im_num,x,y] += 1 / math.sqrt( ( x - position[0] )**2 + ( y - position[1] )**2 )
                    else:
                        distance_map[im_num,x,y] = 1 / distance_map[im_num,x,y]

    return(power_map, density_map, distance_map, IRDrop_map)

#---------- end - Function ----------


#---------- Execution ----------

def main(argv=[], setting={}):

    setting['reload_model'] = True

    reload_model = setting.get('reload_model')
    image_height = setting.get('image_height')
    image_width  = setting.get('image_width')
    circut_height = setting.get('circut_height')
    circut_width = setting.get('circut_width')
    epochs = setting.get('epochs')

    # Create an instance of the model
    model = autoencoder()
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.98,
            staircase=True,
        )
    model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='mse',
            metrics=['mse', 'mae', 'mape'],
        )

    # Reload an existing model
    if( reload_model ):
        model.load_weights("model/model_weights")
        with open("model/parameters","r") as f:
            max_power = float(f.readline())
            max_distance = float(f.readline())
            max_IRDrop = float(f.readline())

    # Evaluate the model

    power_map, density_map, distance_map, IRDrop_map = read_file("data/evaluating/*", setting)
    power_map = power_map/max_power
    distance_map = distance_map/max_distance
    IRDrop_map = IRDrop_map/max_IRDrop
    power_map = power_map[...,np.newaxis]
    IRDrop_map = IRDrop_map[...,np.newaxis]
    power_map = Concatenate()([
            power_map,
            density_map[...,np.newaxis],
            distance_map[...,np.newaxis],
        ])

    y_pred = model.predict(power_map)

    # Display the result

    fig, axes = plt.subplots(4, 4)
    plt.subplots_adjust(hspace=0.4)
    denorm_pred_IRDrop = y_pred# * max_IRDrop
    denorm_IRDrop  = IRDrop_map# * max_IRDrop

    i = 0
    for line in range(4):
        for row in range(0,4,2):
            a = np.squeeze(denorm_pred_IRDrop[i,...])
            b = np.squeeze(denorm_IRDrop[i,...])
            axes[line,row].axis('off')
            im = axes[line,row].imshow(a, cmap="jet")
            axes[line,row+1].axis('off')
            im = axes[line,row+1].imshow(b, cmap="jet")
            axes[line,row].title.set_text("range: {:.4f}\nmax: {:.4f}  min:{:.4f}".format(np.max(a)-np.min(a),np.max(a),np.min(a)))
            axes[line,row+1].title.set_text("range: {:.4f}\nmax: {:.4f}  min:{:.4f}".format(np.max(b)-np.min(b),np.max(b),np.min(b)))
            i += 1

    err = abs( denorm_pred_IRDrop - denorm_IRDrop )

    print()
    print()
    print("Mean Error:", np.mean(err))
    print("RMSE:", np.sqrt(np.mean(err*err)))
    print("Max Error:", np.max(err))
    print()
    print()

    plt.savefig('report.png', transparent=True, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main(sys.argv[1:], setting)

#---------- end - Execution ----------
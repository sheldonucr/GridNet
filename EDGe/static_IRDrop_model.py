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
        'reload_model': True,
        'image_height': 32,
        'image_width': 32,
        'circut_height': 200000,
        'circut_width': 200000,
        'epochs': 1800,
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

    if( '-r' in argv ):
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

    # Train a new model
    else:

        power_map_train, density_map_train, distance_map_train, IRDrop_map_train = read_file("data/training/*", setting)
        max_power = np.max(power_map_train)
        max_distance = np.max(distance_map_train)
        max_IRDrop = np.max(IRDrop_map_train)
        power_map_train = power_map_train/max_power
        distance_map_train = distance_map_train/max_distance
        IRDrop_map_train = IRDrop_map_train/max_IRDrop
        power_map_train = power_map_train[...,np.newaxis]
        IRDrop_map_train = IRDrop_map_train[...,np.newaxis]
        power_map_train = Concatenate()([
                power_map_train,
                density_map_train[...,np.newaxis],
                distance_map_train[...,np.newaxis],
            ])

        power_map_test, density_map_test, distance_map_test, IRDrop_map_test = read_file("data/testing/*", setting)
        power_map_test = power_map_test/max_power
        distance_map_test = distance_map_test/max_distance
        IRDrop_map_test = IRDrop_map_test/max_IRDrop
        power_map_test = power_map_test[...,np.newaxis]
        IRDrop_map_test = IRDrop_map_test[...,np.newaxis]
        power_map_test = Concatenate()([
                power_map_test,
                density_map_test[...,np.newaxis],
                distance_map_test[...,np.newaxis],
            ])

        train_ds = tf.data.Dataset.from_tensor_slices((power_map_train, IRDrop_map_train)).batch(1)
        test_ds = tf.data.Dataset.from_tensor_slices((power_map_test, IRDrop_map_test)).batch(1)

        start_time = time()
        history = model.fit(train_ds, epochs=epochs,
                validation_data=test_ds,
                validation_freq=1,
            )
        end_time = time()
        used_time = end_time-start_time
        print("Elapsed time: %03d:%02d:%05.2f"%(int(used_time/3600),int(used_time/60)%60,used_time%60))

        # Save the model
        model.save_weights("model/model_weights", save_format='tf')
        with open("model/parameters","w") as f:
            f.write(str(max_power)+"\n")
            f.write(str(max_distance)+"\n")
            f.write(str(max_IRDrop)+"\n")

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

    fig, axes = plt.subplots(2, 3)
    denorm_pred_IRDrop = np.squeeze( y_pred[0,...] ) * max_IRDrop
    denorm_IRDrop  = np.squeeze(IRDrop_map[0,...]) * max_IRDrop
    max_IRDrop_im = max(np.max(denorm_IRDrop), np.max(denorm_pred_IRDrop))
    min_IRDrop_im = np.min(denorm_IRDrop)
    min_display_IRDrop_im = min_IRDrop_im
    err = abs(denorm_pred_IRDrop - denorm_IRDrop)
    norm = Normalize(vmin = min_display_IRDrop_im, vmax = max_IRDrop_im)
    im = axes[0,0].imshow(power_map[0,...,0], cmap="jet")
    axes[0,1].axis('off')
    im = axes[0,2].imshow(distance_map[0,...], cmap="jet")
    im = axes[1,0].imshow(denorm_pred_IRDrop, norm=norm, cmap="jet")
    im = axes[1,1].imshow(err, cmap="jet")
    axes[1,1].axis('off')
    im = axes[1,2].imshow(denorm_IRDrop, norm=norm, cmap="jet")
    axes[1,0].title.set_text("Predicted IR Map")
    axes[1,1].title.set_text("Error")
    axes[1,2].title.set_text("Ground Truth")
    print("Mean Error:", np.mean(err))
    print("RMSE:", np.sqrt(np.mean(err*err)))
    print("Max Error:", np.max(err))
    print("Predicted Range: (", np.min(denorm_pred_IRDrop), ",", np.max(denorm_pred_IRDrop), ")")
    print("Real Range: (", np.min(denorm_IRDrop), ",", np.max(denorm_IRDrop), ")")
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig('result.png', transparent=True, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main(sys.argv[1:], setting)

#---------- end - Execution ----------
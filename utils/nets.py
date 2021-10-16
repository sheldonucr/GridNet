import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

class G_conv_resist_voltage(object):
    def __init__(self, input_imgsize):
        self.name = 'G_conv_resist_voltage'
        self.input_imgsize = input_imgsize
        self.batch_norm_flag = True

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            if self.input_imgsize == 32:
                bottleneck_idx = 4
                filters_list = [64,128,256,512, 512, 512,256,128,64,1]
                strides_list = [2,2,2,2, 1, 2,2,2,1,2]
                kernels_list = [5,5,5,3, 1, 5,5,5,5,5]
            elif self.input_imgsize == 64:
                bottleneck_idx = 4
                filters_list = [64,128,256,512, 512, 512,256,128,64,1]
                strides_list = [2,2,2,2, 2, 2,2,2,2,2]
                kernels_list = [5,5,5,5, 3, 5,5,5,5,5]
            elif self.input_imgsize == 128:
                bottleneck_idx = 5
                filters_list = [64,128,256,512,512, 512, 512,512,256,128,64,1] # bzx128x128x4 -> bzx64x64x64 -> bzx32x32x128  16x16x256  8x8x512  4x4x512  2x2x512  4x4x512  8x8x512  16x16x256  32x32x128  64x64x64  64x64x1
                strides_list = [2,2,2,2,2, 2, 2,2,2,2,2,2]
                kernels_list = [5,5,5,5,5, 3, 5,5,5,5,5,5]
            else:
                raise Exception(f"The Generater model for input_size={self.input_imgsize} is not implemented!")

            output = z
            for i in range(len(filters_list)):
                if i <= bottleneck_idx: # Encoder
                    output = tf.layers.conv2d(output,filters=filters_list[i], kernel_size=kernels_list[i],
                                strides=strides_list[i], padding='same', activation=None, kernel_initializer=tf.random_normal_initializer(0, 0.02))
                    output = tf.nn.leaky_relu(output, alpha = 0.1)
                else: # Decoder
                    output = tf.layers.conv2d_transpose(output, filters=filters_list[i], kernel_size=kernels_list[i],
                                strides=strides_list[i], padding='same', activation = None, kernel_initializer=tf.random_normal_initializer(0, 0.02))
                    # No activation_fn in the output layer
                    if i < len(filters_list)-1:
                        output = tf.nn.leaky_relu(output, alpha = 0.1)
            
            return output

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
class D_conv_resist_voltage(object):
    def __init__(self, input_imgsize):
        self.name = 'D_conv_resist_voltage'
        self.input_imgsize = input_imgsize
        self.batch_norm_flag = True

    def __call__(self, x, y, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

             
            if self.input_imgsize == 32:
                filters_list = [64,128,256,512]
                strides_list = [2,2,2,2]
                kernels_list = [5,5,5,5]
            elif self.input_imgsize == 64:
                filters_list = [64,128,256,512,1024]
                strides_list = [2,2,2,2,2]
                kernels_list = [5,5,5,5,5]
            elif self.input_imgsize == 128:
                filters_list = [64,128,256,512,1024,1024]
                strides_list = [2,2,2,2,2,2]
                kernels_list = [5,5,5,5,5,5]
            else:
                raise Exception(f"The Discriminator model for input_size={self.input_imgsize} is not implemented!")

            output = tf.concat([x, y], 3)
            for i in range(len(filters_list)):
                output = tf.layers.conv2d(output,filters=filters_list[i], kernel_size=[kernels_list[i], kernels_list[i]],
                            strides=strides_list[i], padding='same', activation=None, kernel_initializer=tf.random_normal_initializer(0, 0.02))
                output = tf.nn.leaky_relu(output, alpha = 0.1)

            output = tcl.flatten(output)

            output = tcl.fully_connected(output, 512, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))

            output = tcl.fully_connected(output, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))

            return output

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)



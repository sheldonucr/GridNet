import tensorflow as tf
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
import logging
import json
from pprint import pprint
os.environ['KMP_DUPLICATE_LIB_OK']='True'


sys.path.append('utils')
from nets import *
from datas import *
from misc import *

from shutil import copyfile

from timeit import default_timer as timer

import re

from datetime import datetime

class Parser(argparse.ArgumentParser):
    def __init__(self): 
        super(Parser, self).__init__(description='Learning surrogate with mixed residual norm loss')
        self.add_argument('--sample-dir', type=str, default="./Samples", help='directory to save output')      
        # data 
        self.add_argument('--test-pct', type=float, default=0.15, help='percentage of data points reserved for testing')
        self.add_argument('--data-dir', type=str, default='/fermi_data/shared/IRGAN/data128/mortal1234/,/fermi_data/shared/IRGAN/data128/immortal1234/', help='delimited data dirs')
        self.add_argument('--train-csv', type=str, default=None, help='path of train data filenames .csv file')
        self.add_argument('--test-csv', type=str, default=None, help='path of test data filenames .csv file')
        self.add_argument('--input-channel', type=int, default=4, help='number of input channels')
        self.add_argument('--input-imgsize', type=int, default=128, help='size of input image')
        # training
        self.add_argument('--epochs', type=int, default=3000, help='number of epochs to train')
        self.add_argument('--batch-size', type=int, default=8, help='input batch size for training')
        self.add_argument('--lr', type=float, default=1e-6, help='learning rate')
        self.add_argument('--l2-alpha', type=float, default=1e2, help='weight of l2 loss in the loss function')
        self.add_argument('--gp-lambda', type=float, default=1e1, help='weight of GP loss in the loss function')
        self.add_argument('--ckpt-freq', type=int, default=1, help='Model is saved every ckpt-freq epochs.')
        
        # inference
        self.add_argument('--is-inference', action='store_true', default=False, help='training or inference')
        self.add_argument('--save-output-csv', action='store_true', default=False, help='save inference results and ground truth in csv format')
        self.add_argument('--is-gradient', action='store_true', default=False, help='output gradient(sensitivity) info')
        self.add_argument('--grad-node-list', type=str, default=None, help='The list of nodal valtages that requires gradient(sensitivity) info')
        self.add_argument('--ckpt-file', type=str, default=None, help='Path to the ckpt file to be loaded. Only valid in inference mode')   

    def parse(self):
        args = self.parse_args()

        # Check if training data path exists
        if args.data_dir == None:
            raise Exception(f"Please provide the path to the training data!")
        else:
            args.data_dir = args.data_dir.split(',')
            for datapath in args.data_dir: 
                if not os.path.exists(datapath):
                    raise Exception(f"Provided training data path {datapath} Not Exists!")
            print(f"Training data path: {args.data_dir}")

        # Create output dir
        dt = datetime.now()
        args.date = dt.strftime("%Y%m%d%H%M%S")
        hparams = f'GridNet_{"Inference" if args.is_inference else "Train"}_Size_{args.input_imgsize}_{args.date}'
        args.run_dir = args.sample_dir + '/' + hparams
        args.ckpt_dir = args.run_dir + '/checkpoints'
        args.code_bkup_dir = args.run_dir + '/code_bkup'
        mkdirs(args.run_dir, args.ckpt_dir, args.code_bkup_dir)

        if args.is_inference and args.is_gradient:
            if args.grad_node_list is None:
                raise Exception(f"Missing grad-node-list file! CSV file of nodal voltages which require sensitivity info should be provided through --grad-node-list arg!")
            args.grad_dir = args.run_dir + '/gradients'
            mkdirs(args.grad_dir)

        print('Arguments:')
        pprint(vars(args))
        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)

        return args

class CGAN():
    def __init__(self, generator, discriminator, data, args):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.args = args
        # self.batch_size = self.args.batch_size
        self.alpha = self.args.l2_alpha
        self.lambda_ = self.args.gp_lambda 

        # data
        self.size = self.args.input_imgsize
        self.channel = self.args.input_channel
        
        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel]) # [r_col r_row I time] bz x sz x sz x 4 
        self.y = tf.placeholder(tf.float32, shape=[None, self.size, self.size, 1]) # nodal voltage bz x sz x sz x 1
        self.batch_size = tf.shape(self.X)[0]

        # nets
        self.G_sample = self.generator(self.X)

        self.D_real = self.discriminator(self.y, self.X)
        self.D_fake = self.discriminator(self.G_sample, self.X, reuse = True)
        
        # loss
        self.wgan_d_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)

        self.L2_loss = tf.sqrt(tf.reduce_mean(tf.losses.mean_squared_error(self.y, self.G_sample)) + 1e-10)
        self.G_loss = - tf.reduce_mean(self.D_fake) + self.alpha * self.L2_loss

        # gradient penalty
        self.gp_loss = self.gradient_penalty()
        self.D_loss = self.wgan_d_loss + self.lambda_ * self.gp_loss

        # solver
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_g = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G_conv_current_stress')
        update_ops_d = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D_conv_current_stress')
        with tf.control_dependencies(update_ops_d):
            self.D_solver = tf.train.AdamOptimizer(learning_rate=self.args.lr).minimize(self.D_loss, var_list=self.discriminator.vars)

        with tf.control_dependencies(update_ops_g):
            self.G_solver = tf.train.AdamOptimizer(learning_rate=self.args.lr).minimize(self.G_loss, var_list=self.generator.vars)

        self.check_op = tf.add_check_numerics_ops()
        self.summary = self.add_summary()

        # only save trainable and bn variables
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=2)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, run_dir, ckpt_dir, training_epoches, ckpt_freq):
        fig_count = 0
        Global_loss = np.empty((1,2))
        self.sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(ckpt_dir+'/summary', graph=self.sess.graph)

        start_epoch = 0
        # use check point as the starter of iteration
        # self.saver.restore(self.sess, 'ckpt/heatmap_wgan_cifar10like_power_2_heat_with_L2norm_loss_alpha_0.1.ckpt-104000')
        # start_epoch = 104001
        
        for epoch in range(start_epoch, training_epoches):
            print(f'>Start traning epoch {epoch}')
            D_loss_total = 0
            G_loss_total = 0
            for it in range(self.data.num_batch):
                # update D
                # tmr_start = timer()
                X_b,y_b,filenames = self.data()
                # tmr_end = timer()
                # print(f'Data loading time: {tmr_end-tmr_start}')

                # tmr_start = timer()
                try:
                    self.sess.run(
                        [self.D_solver, self.check_op],
                        feed_dict={self.X: X_b, self.y: y_b}
                        )
                    # update G
                    k = 2 # train G k times in each iteration so that G can catch up with D
                    for _ in range(k):
                        self.sess.run(
                            [self.G_solver, self.check_op],
                            feed_dict={self.X: X_b, self.y: y_b}
                        )
                except Exception as e:
                    logging.warning('error occured!!!!! {}'.format(e))
                    print(e.op.inputs[0].eval(session=self.sess,feed_dict={self.X: X_b, self.y: y_b}))
                    raise Exception(f"Training process terminated with error!")
                # tmr_end = timer()
                # print(f'Model training time: {tmr_end-tmr_start}')

                # tmr_start = timer()
                # Loss calculation
                D_loss_curr = self.sess.run(
                        self.D_loss,
                        feed_dict={self.X: X_b, self.y: y_b})
                G_loss_curr = self.sess.run(
                        self.G_loss,
                        feed_dict={self.X: X_b, self.y: y_b})
                D_loss_total += D_loss_curr
                G_loss_total += G_loss_curr
                # tmr_end = timer()
                # print(f'Loss calc timer: {tmr_end-tmr_start}')

                if it % 10 == 0 or it == self.data.num_batch:
                    print(f' Finished training for batch {it}/{self.data.num_batch}.', end = "\r")

            # Loss calculation after each epoch
            D_loss_total /= self.data.num_batch
            G_loss_total /= self.data.num_batch
            print(' Epoch: {}, D_loss: {:.6}, G_loss: {:.6}'.format(epoch, D_loss_total, G_loss_total))
            Global_loss = np.vstack((Global_loss, [D_loss_total, G_loss_total]))
            np.savetxt('{}/global_loss.csv'.format(run_dir), Global_loss, delimiter=",")

            # Tensor board logging
            x_test,y_test,_ = self.data.fetch_test(8)
            feed_d = {self.X: x_test, self.y: y_test}
            summary = self.sess.run(self.summary, feed_dict=feed_d)
            summary_writer.add_summary(summary, global_step = epoch)
            summary_writer.flush()

            # Plot training data
            samples, D_logit_fake, D_logit_real  = self.sess.run(
                                                [self.G_sample, self.D_fake, self.D_real],
                                                feed_dict={self.X: X_b, self.y: y_b})

            fig = data2fig(samples, y_b, filenames, D_logit_fake, D_logit_real)
            plt.savefig('{}/{}_train.png'.format(run_dir, str(epoch).zfill(4)), bbox_inches='tight')
            plt.close(fig)

            # Plot test data
            x_test,y_test,filenames = self.data.fetch_test(8)
            samples, D_logit_fake, D_logit_real  = self.sess.run(
                                                    [self.G_sample, self.D_fake, self.D_real],
                                                    feed_dict={self.X: x_test, self.y: y_test})

            fig = data2fig(samples, y_test, filenames, D_logit_fake, D_logit_real)
            plt.savefig('{}/{}_test.png'.format(run_dir, str(epoch).zfill(4)), bbox_inches='tight')
            plt.close(fig)

            # Save model
            if epoch % ckpt_freq == 0 and epoch != 0:
                self.saver.save(self.sess, os.path.join(ckpt_dir, "GridNet.ckpt"), global_step = epoch)
                print(' Model saved')

    def inference(self, run_dir, ckpt_file, test_csv, is_gradient, save_output_csv):
        self.sess.run(tf.global_variables_initializer())

        # use check point as the starter of iteration
        self.saver.restore(self.sess, ckpt_file)

        # gradient calculation part
        if is_gradient:
            G_sample_split = tf.split(tf.reshape(self.G_sample,[self.batch_size ,self.size*self.size]), self.size*self.size, axis = 1) # bzxszxszx1 is split into a list of [bzx1] with length of sz^2
            idx_placeholder = tf.placeholder(tf.int32, shape=(None,))
            selected_node_voltage = tf.gather(G_sample_split, indices = idx_placeholder, axis = 0, batch_dims = 0) # pick one node voltage [bzx1] out of sz^2 nodal voltages
            grad_operation = tf.gradients(selected_node_voltage,self.X)[0]

            gradient_list = [int(row[0]) for row in list(csv.reader(open(self.args.grad_node_list, "r"), delimiter=","))] # convert to list

        reader = csv.reader(open(test_csv, "r"), delimiter=",")
        dataset = list(reader)
        dataset = np.array(dataset)
        dataset = dataset.reshape(-1).tolist()

        bz = 8
        count = 0
        fig_count = 1
        test_filenames = []
        norm_test = []
        while count < len(dataset):
            if (count+bz) > len(dataset):
                path_list = dataset[count:]
                count = len(dataset)
            else:
                path_list = dataset[count : count+bz]
                count += bz
            
            # path_list = []
            # while len(path_list) < 8 :
            #     data_path = dataset[count]
            #     path_list.append(data_path)
            #     count += 1
            #     if (count % 100 == 0):
            #         print(f'Finished inference for {count} points.')
            #     if (count == len(dataset)):
            #         end = True
            #         # for i in range(8 - len(path_list)):
            #         #     path_list.append(data_path)

            # Fetch [r_col r_row I time]
            x_resist = [get_resist(data_path, self.size) for data_path in path_list]
            x_resist = np.array(x_resist).astype(np.float64)

            # Fetch node voltage
            y_voltage = [get_voltage(data_path, self.size) for data_path in path_list]
            y_voltage = np.array(y_voltage).astype(np.float64)

            filenames = path_list
            filenames = [re.findall('r[0-9]+_[0-9]+_[0-9]+',data_path)[0] for data_path in filenames]
            test_filenames.extend(filenames)

            if is_gradient:
                for k in gradient_list:
                    if k > self.size*self.size:
                        raise ValueError(f'The nodel voltage number in --grad-node-list is out of range! Allowed range is 1-{self.size*self.size} but {k} is provided. Please check your --grad-node-list file!')
                    # tmr_start = timer()
                    grads = self.sess.run(grad_operation, feed_dict={self.X: x_resist, self.y: y_voltage, idx_placeholder: np.array([k-1])})
                    # grads, sample_split, nodevoltage = self.sess.run([grad_operation,G_sample_split,selected_node_voltage], feed_dict={self.X: x_resist, self.y: y_voltage, idx_placeholder: np.array([k-1])}) # debug code
                    # tmr_end = timer()
                    # print(tmr_end - tmr_start)
                    for l in range(x_resist.shape[0]):
                        grad_sample = grads[l,:,:,0]
                        np.savetxt('{}/{}_{}_r_col_gradient.csv'.format(self.args.grad_dir, filenames[l], str(k).zfill(4)), grad_sample, delimiter=",", fmt='%s')

                        grad_sample = grads[l,:,:,1]
                        np.savetxt('{}/{}_{}_r_row_gradient.csv'.format(self.args.grad_dir, filenames[l], str(k).zfill(4)), grad_sample, delimiter=",", fmt='%s')

            # tmr_start = timer()
            samples = self.sess.run(self.G_sample, feed_dict={self.X: x_resist, self.y: y_voltage})
            # tmr_end = timer()
            # print(tmr_end - tmr_start)

            fig = data2fig(samples, y_voltage, filenames)
            plt.savefig('{}/{}_inference.png'.format(run_dir, str(fig_count).zfill(4)), bbox_inches='tight')
            plt.close(fig)
            fig_count += 1

            # Error calculation
            for i in range(samples.shape[0]):
                sample = samples[i,:,:,0]
                truth = y_voltage[i,:,:,0]

                # convert voltage from [-1 1] back to [1.5 1.8]
                truth = (truth + 1) / 2 * (1.8-1.5) + 1.5
                sample = (sample + 1) / 2 * (1.8-1.5) + 1.5

                if save_output_csv:
                    np.savetxt('{}/{}_predicted.csv'.format(run_dir, filenames[i]), sample, delimiter=",", fmt='%s')
                    np.savetxt('{}/{}_ground_truth.csv'.format(run_dir, filenames[i]), truth, delimiter=",", fmt='%s')

                # RMSE calculation
                norm_test.append(np.sqrt(np.power((sample - truth), 2).sum()/(sample.shape[0]**2)))

            if (count % (1*bz) == 0 or count == len(dataset)):
                print(f'Finished inference for {count}/{len(dataset)} points.', end = "\r")

        test_filenames_np = np.array(test_filenames).reshape((len(test_filenames),1))
        norm_test_np = np.array(norm_test).reshape((len(norm_test),1))
        test_output = np.hstack((test_filenames_np,norm_test_np))
        
        np.savetxt(run_dir+'/norm_test.csv', test_output, delimiter=",", fmt='%s')
    
    def gradient_penalty(self):
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        differences = self.G_sample - self.y
        interpolates = self.y + (alpha * differences)
        gradients = tf.gradients(self.discriminator(interpolates, self.X, reuse = True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients + 1e-10), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return gradient_penalty

    def add_summary(self):
        """Add summary operation.

        Return
        ------
        summary_op: tf summary.
        """

        tf.summary.scalar('G_loss', self.G_loss)
        tf.summary.scalar('D_loss', -self.D_loss)
        tf.summary.scalar('wgan_d_loss', -self.wgan_d_loss)
        tf.summary.scalar('gp_loss', self.gp_loss)
        tf.summary.scalar('L2_loss', self.L2_loss)

        tf.summary.image('real', self.y, max_outputs=8)
        tf.summary.image('fake', self.G_sample, max_outputs=8)
        tf.summary.image('input_current', self.X, max_outputs=8)

        summary_op = tf.summary.merge_all()

        return summary_op


if __name__ == '__main__':
    args = Parser().parse()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # backup the code of current model
    code_path_src = os.path.basename(__file__)
    code_path_dst = args.code_bkup_dir + '/' + code_path_src
    copyfile(code_path_src,code_path_dst)
    mkdirs(args.code_bkup_dir + '/utils')
    copyfile('utils/datas.py', args.code_bkup_dir + '/utils/datas.py')
    copyfile('utils/nets.py', args.code_bkup_dir + '/utils/nets.py')
    copyfile('utils/misc.py', args.code_bkup_dir + '/utils/misc.py')

    # Nets and Datas
    generator = G_conv_resist_voltage(args.input_imgsize)
    discriminator = D_conv_resist_voltage(args.input_imgsize)

    if args.is_inference:
        data=None
    else:
        data = resist_voltage(args)

    # run
    cgan = CGAN(generator, discriminator, data, args)
    if args.is_inference:
        cgan.inference(args.run_dir, args.ckpt_file, args.test_csv, args.is_gradient, args.save_output_csv)
    else:
        cgan.train(args.run_dir, args.ckpt_dir, args.epochs, args.ckpt_freq)


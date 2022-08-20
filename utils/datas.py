import os,sys
import glob
import numpy as np
import csv
import re
import math

# [r_col r_row I time] bz x 128 x 128 x 4 
def get_resist(data_path, size):
    resist = np.zeros([size, size, 4])

    # typical data_path: r6_2_10
    time = data_path[data_path.rfind('_')+1:-4]
    time = int(time) / 10
    time = time * np.ones([size,size])

    current_path = data_path[:data_path.rfind('r')]+'i'+data_path[data_path.rfind('r')+1:data_path.rfind('_')]+'_0.txt' # replace r with i, replace time with 0
    current = np.genfromtxt(current_path, dtype=np.float64)[:size*size]
    current = (current-0)/(70e-5-0) # convert current to [0 1]
    current = 2 * current - 1 # convert y to [-1 1]
    current = np.transpose(current.reshape([size,size]))

    x = np.genfromtxt(data_path, dtype=np.float64)
    x = (x-0.1)/(1.0-0.1) # convert x from [0.1 1.0] to [0 1]
    x = 2 * x - 1 # convert x to [-1 1]
    x_col = np.transpose(x[:(size-1)*size].reshape([size,(size-1)])) # 127 x 128, col resistor matrix
    x_row = x[size*(size-1):].reshape([size,(size-1)]) # 128 x 127, row resistor matrix

    resist[:(size-1),:,0] = x_col
    resist[:,:(size-1),1] = x_row
    resist[:,:,2] = current
    resist[:,:,3] = time

    return resist

# nodal voltage bz x 128 x 128 x 1
def get_voltage(data_path, size):
    
    # data_path = data_path[:data_path.rfind('_')]+'.txt' # remove current value 
    data_path = data_path[:data_path.rfind('r')]+'v'+data_path[data_path.rfind('r')+1:] # replace r with v

    voltage = np.zeros([size, size, 1])

    y = np.genfromtxt(data_path, dtype=np.float64)[:size*size]
    y = (y - 1.5) / (1.8-1.5) # convert y from [1.5 1.8] to [0 1]
    y = 2 * y - 1 # convert y to [-1 1]
    y = np.transpose(y.reshape([size,size]))

    voltage[:,:,0] = y

    return voltage

class resist_voltage():
    def __init__(self, args):
        self.args = args
        self.test_pct = self.args.test_pct
        self.size = self.args.input_imgsize
        self.channel = self.args.input_channel
        self.data_dir = self.args.data_dir
        self.batch_size = self.args.batch_size

        if args.train_csv is not None and args.test_csv is not None:
            # read dataset from csv
            print(f'>Reading training and test data filenames from {args.train_csv} and {args.test_csv}')
            # reader = csv.reader(open('Samples/grid_gan_128x128_20200524/train_data.csv', "r"), delimiter=",")
            reader = csv.reader(open(args.train_csv, "r"), delimiter=",")
            self.train_data = list(reader)
            np.random.shuffle(self.train_data)

            # reader = csv.reader(open('Samples/grid_gan_128x128_20200524/test_data.csv', "r"), delimiter=",")
            reader = csv.reader(open(args.test_csv, "r"), delimiter=",")
            self.test_data = list(reader)
            np.random.shuffle(self.test_data)

        else:
            print(f'>Generating training and test dataset by iterating {self.data_dir}')
            # iterate data file names into list
            self.all_data_file = []
            for datapath in self.data_dir:
                self.all_data_file.extend(glob.glob(datapath + 'r*.txt'))
            # self.all_data_file = self.all_data_file[0:737]

            # extract unique design names
            self.all_design_file = [re.findall('r[0-9]+',file_name)[0] for file_name in self.all_data_file]
            self.all_design_file = list(set(self.all_design_file))

            # random permutation of the design names
            np.random.shuffle(self.all_design_file)

            # split design names into train set and test set
            split_index = int(len(self.all_design_file)*(1-self.test_pct))
            self.train_design = self.all_design_file[:split_index]
            self.test_design = self.all_design_file[split_index:]

            # random permutation of the data file names
            np.random.shuffle(self.all_data_file)

            # split data file names into training set and test set
            self.train_data = [data for data in self.all_data_file if re.findall('r[0-9]+',data)[0] in self.train_design]
            self.test_data = [data for data in self.all_data_file if re.findall('r[0-9]+',data)[0] in self.test_design]

            # save test and training dataset to run dir
            np.savetxt(args.run_dir+'/train_data.csv', self.train_data, delimiter=",", fmt='%s')
            np.savetxt(args.run_dir+'/test_data.csv', self.test_data, delimiter=",", fmt='%s')

        self.batch_count = 0
        self.valid_data_count = 0
        self.len_train = len(self.train_data)
        self.len_test = len(self.test_data)
        self.num_batch = int(self.len_train/self.batch_size)

        print(f' Number of training data points: {self.len_train}')
        print(f' Number of test data points: {self.len_test}')
        print(f' Number of batches in each epoch: {self.num_batch}')

    def __call__(self):
        # check if last batch is reached in each epoch
        path_list = self.train_data[self.batch_count*self.batch_size : (self.batch_count+1)*self.batch_size]
        self.batch_count += 1
        if self.batch_count == self.num_batch:
            self.batch_count = 0
            np.random.shuffle(self.train_data)

        # Generate dataset names for plot use
        filenames = [re.findall('r[0-9]+_[0-9]+_[0-9]+',data_path)[0] for data_path in path_list]

        # Fetch input: [r_col r_row I time] bz x size x size x 4 
        x_resist = [get_resist(data_path, self.size) for data_path in path_list]
        x_resist = np.array(x_resist).astype(np.float64)

        # Fetch output: node voltage
        y_voltage = [get_voltage(data_path, self.size) for data_path in path_list]
        y_voltage = np.array(y_voltage).astype(np.float64)

        return x_resist, y_voltage, filenames

    def fetch_test(self, n_sample):
        path_list = np.random.choice(self.test_data, n_sample, replace = False).tolist()

        # Generate dataset names for plot use
        filenames = [re.findall('r[0-9]+_[0-9]+_[0-9]+',data_path)[0] for data_path in path_list]

        # Fetch current density data as x
        x_test = [get_resist(data_path, self.size) for data_path in path_list]
        x_test = np.array(x_test).astype(np.float32)

        # Fetch stress data as y
        y_test = [get_voltage(data_path, self.size) for data_path in path_list]
        y_test = np.array(y_test).astype(np.float32)

        return x_test, y_test, filenames


    # def data2fig(self, samples, truths, filenames, D_logit_fake, D_logit_real):
    #     n_batch = samples.shape[0]
    #     n_row = math.ceil(n_batch/2)
    #     n_col = 4
    #     fig = plt.figure(figsize=(10*n_col, 10*n_row))
    #     gs = gridspec.GridSpec(n_row, n_col)
    #     gs.update(wspace=0.1, hspace=0.4)

    #     norm = []
    #     for i in range(n_batch):
    #         sample = samples[i,:,:,0]
    #         truth = truths[i,:,:,0]

    #         # convert voltage from [-1 1] back to [1.5 1.8]
    #         truth = (truth + 1) / 2 * (1.8-1.5) + 1.5
    #         sample = (sample + 1) / 2 * (1.8-1.5) + 1.5

    #         # RMSE calculation
    #         norm.append(np.sqrt(np.power((sample - truth), 2).sum()/(sample.shape[0]**2)))

    #         # Plot predicted result
    #         ax = plt.subplot(gs[2*i])
    #         plt.axis('off')
    #         ax.set_xticklabels([])
    #         ax.set_yticklabels([])
    #         ax.set_aspect('equal')
    #         ax.set_title('Range: {0:.4f}\nfake: {3:.3f}  real: {4:.3f}\nmax: {1:.4f}  min: {2:.4f}'.format(sample.max()-sample.min(), sample.max(), sample.min(), D_logit_fake[i,0], D_logit_real[i,0]))
    #         plt.imshow(sample, cmap='hot', interpolation='nearest')

    #         # Plot ground truth
    #         ax = plt.subplot(gs[2*i+1])
    #         plt.axis('off')
    #         ax.set_xticklabels([])
    #         ax.set_yticklabels([])
    #         ax.set_aspect('equal')
    #         ax.set_title('{0}: {1}\nNorm: {2:.2f} Range: {3:.4f}\nmax: {4:.4f}  min: {5:.4f}'.format(i, filenames[i], norm[i], truth.max()-truth.min(), truth.max(), truth.min()))
    #         plt.imshow(truth, cmap='hot', interpolation='nearest')

    #     rmse = np.sqrt(np.square(norm).mean())
    #     fig.suptitle('RMSE={}'.format(rmse))
    #     return fig


if __name__ == '__main__':
    batch_size = 2
    data = resist_voltage()
    print(data(2)[0].shape)
    X_b,y_b,filenames = data.fetch_test(2)
    # z_b = [int(data_path[data_path.find('_')+1:]) for data_path in filenames]
    # z_b = np.array(z_b) / 10
    # z_b = z_b.reshape(batch_size,1)

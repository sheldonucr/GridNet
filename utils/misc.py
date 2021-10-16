import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
plt.switch_backend('agg')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(*paths):
    if isinstance(paths, list) or isinstance(paths, tuple):
        for path in paths:
            mkdir(path)
    else:
        raise ValueError

def save_stats(save_dir, logger, *metrics):
    for metric in metrics:
        metric_list = logger[metric]
        np.savetxt(save_dir + f'/{metric}.txt', metric_list)
        # plot stats
        metric_arr = np.loadtxt(save_dir + f'/{metric}.txt')
        if len(metric_arr.shape) == 1:
            metric_arr = metric_arr[:, None]
        lines = plt.plot(range(1, len(metric_arr)+1), metric_arr)
        labels = [f'{metric_arr[-5:, i].mean():.4f}' for i in range(metric_arr.shape[-1])]
        plt.legend(lines, labels)
        plt.savefig(save_dir + f'/{metric}.pdf')
        plt.close()

def data2fig(samples, truths, filenames, D_logit_fake=None, D_logit_real=None):
    n_batch = samples.shape[0]
    n_row = math.ceil(n_batch/2)
    n_col = 4
    fig = plt.figure(figsize=(10*n_col, 10*n_row))
    gs = gridspec.GridSpec(n_row, n_col)
    gs.update(wspace=0.1, hspace=0.4)

    norm = []
    for i in range(n_batch):
        sample = samples[i,:,:,0]
        truth = truths[i,:,:,0]

        # convert voltage from [-1 1] back to [1.5 1.8]
        truth = (truth + 1) / 2 * (1.8-1.5) + 1.5
        sample = (sample + 1) / 2 * (1.8-1.5) + 1.5

        # RMSE calculation
        norm.append(np.sqrt(np.power((sample - truth), 2).sum()/(sample.shape[0]**2)))

        # Plot predicted result
        ax = plt.subplot(gs[2*i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if D_logit_fake is not None and D_logit_real is not None:
            ax.set_title('Range: {0:.4f}\nfake: {3:.3f}  real: {4:.3f}\nmax: {1:.4f}  min: {2:.4f}'.format(sample.max()-sample.min(), sample.max(), sample.min(), D_logit_fake[i,0], D_logit_real[i,0]))
        else:
            ax.set_title('Range: {0:.4f}\nmax: {1:.4f}  min: {2:.4f}'.format(sample.max()-sample.min(), sample.max(), sample.min()))
        plt.imshow(sample, cmap='hot', interpolation='nearest')

        # Plot ground truth
        ax = plt.subplot(gs[2*i+1])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.set_title('{0}: {1}\nNorm: {2:.2f} Range: {3:.4f}\nmax: {4:.4f}  min: {5:.4f}'.format(i, filenames[i], norm[i], truth.max()-truth.min(), truth.max(), truth.min()))
        plt.imshow(truth, cmap='hot', interpolation='nearest')

    rmse = np.sqrt(np.square(norm).mean())
    fig.suptitle('RMSE={}'.format(rmse))
    return fig

    


if __name__ == '__main__':
    print('TODO')


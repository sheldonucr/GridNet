# GridNet

GridNet is a GAN-based IRDrop prediction tool for power grid

- Input: image-like multi-channel tensor, e.g. 4-channel input consisting column resistance, row resistance, current source and time.
- Output: single-channel IRDrop map with the same size of input.
- Can also provide sensitivity(gradient) info for specified nodes. Please refer to the Examples section.

32x32 GridNet Train Process |
:-----:|
<img align="middle" src="./assets/train_process.gif" alt="train_process"  width=460/> |


## Installation

GridNet requires TensorFlow 1.x to be installed as backend. It was tested on Fermi server in Anaconda virtual env with following dependencies:

- python=3.7.3
- tensorflow-gpu=1.14.0
- cudatoolkit=10.1.168
- numpy
- matplotlib

## Examples


- Train 32x32 GridNet:

```
$ python gridnet.py --input-imgsize 32 --input-channel 4 --data-dir /fermi_data/shared/IRGAN/data32/mortal3000/,/fermi_data/shared/IRGAN/data32/mortal6000/,/fermi_data/shared/IRGAN/data32/mortal9000/,/fermi_data/shared/IRGAN/data32/mortal12000/,/fermi_data/shared/IRGAN/data32/immortal3000/,/fermi_data/shared/IRGAN/data32/immortal6000/,/fermi_data/shared/IRGAN/data32/immortal9000/,/fermi_data/shared/IRGAN/data32/immortal12000/ --epochs 100 --batch-size 8 --lr 0.000001
```

- Inference 32x32 GridNet:
Paths of all test data must be provided through `--test-csv`.

```
$ python gridnet.py --input-imgsize 32 --input-channel 4 --is-inference --ckpt-file /fermi_data/wjin/GridNet/Samples/GridNet_Train_Size_32_20211013164554/checkpoints/GridNet.ckpt-99 --test-csv /fermi_data/wjin/GridNet/Samples/GridNet_Train_Size_32_20211013164554/test_data.csv
```

- Inference 32x32 GridNet with both sensitivities and raw outputs saved into .csv files:
Set `--is-gradient` will enable the sensitivity(gradient) calculation. A .csv file which specifies which nodal voltages' sensitivity are to be calculated must be provided through `--grad-node-list`.

```
$ python gridnet.py --input-imgsize 32 --input-channel 4 --is-inference --ckpt-file /fermi_data/wjin/GridNet/Samples/GridNet_Train_Size_32_20211013164554/checkpoints/GridNet.ckpt-99 --test-csv /fermi_data/wjin/GridNet/Samples/GridNet_Train_Size_32_20211013164554/test_data.csv --save-output-csv --is-gradient --grad-node-list ./sensitivity.csv
```

## Note
To run the old version used in original Gridnet ICCAD'20 paper, please checkout the old eb1fc20 version using:
```
git checkout eb1fc208a6f7f3449fd86271997d03bf2b94609c
```
The old version could support only one current source while the later versions (starting from `c9c78da9c93e9f0a0e641d5db0b515300f87e392`) support a curernt source grid.

## Publications

```
@inproceedings{ZhouJin:ICCAD'20,
  author = {Zhou, H. and Jin, W. and Tan, S. X.-D.},
  title = {{GridNet: Fast Data-Driven EM-Induced IR Drop Prediction and Localized Fixing for On-Chip Power Grid Networks}},
  booktitle = {{Proceedings of the 39th International Conference on Computer-Aided Design}},
  series = {ICCAD '20},
  year = {2020},
  month = nov,
  pages = {1--9},
  location = {Virtual Event},
}
```

## The Team

GridNet was originally developed by [Wentian Jin](https://vsclab.ece.ucr.edu/people/wentian-jin) and [Han Zhou](https://vsclab.ece.ucr.edu/people/han-zhou) at [VSCLAB](https://vsclab.ece.ucr.edu/VSCLAB) under the supervision of Prof. [Sheldon Tan](https://profiles.ucr.edu/app/home/profile/sheldont).

GridNet is currently maintained by [Wentian Jin](https://vsclab.ece.ucr.edu/people/wentian-jin).


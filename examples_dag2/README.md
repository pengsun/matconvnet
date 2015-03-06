# Examples for the use of DAG/GTN network 

Demonstrate how to build the DAG/GTN network by combining the Node Data (`n_data.m`) and the Transformers (`tf_xxx`) for your own task, i.e., how to wrap the APIs in the directory `./matlab_dag`. The GTN can be as simple as 
the common feed forward neural network, or as complicated as the TODO. The wrapping we demonstrate in this directory is similar to how `./examples/cnn_train.m` and `./matlab/vl_simplenn.m` wrap the `vl_nnxxx` APIs. 

## Desgin Concept
- The whole DAG is viewed as a big transformer, derived from `tfs_i`, i.e., a DAG (transformer) can be the composition of many small DAGs (transformers). 
- Explicit CPU version or GPU version
- The net `convdag` is thin wrapper of the DAG, managing the training and testing

## Purposes of the Examples
It is suggested that the following examples are read sequentially:

1. `tfw_xpu_lenetDropout`, `mnist_small_tr_xpu_lenetDropout`: the "LeNet" for mnist dataset, and its caller. `xpu` means CPU or GPU version. 
2. `tfw_xpu_lenetTriCon`, `mnist_small_xpu_lenetTriCon`: the modified "LeNet" with triangular connection at the second last layer, and its caller. `xpu` means CPU or GPU version.


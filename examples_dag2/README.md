# Examples for the use of DAG/GTN network 

Demonstrate how to build the DAG/GTN network by combining the Node Data (`n_data.m`) and the Transformers (`tf_xxx`) for your own task, i.e., how to wrap the APIs in the directory `./matlab_dag`. The GTN can be as simple as 
the common feed forward neural network, or as complicated as the TODO. The wrapping we demonstrate in this directory is similar to how `./examples/cnn_train.m` and `./matlab/vl_simplenn.m` wrap the `vl_nnxxx` APIs. 

## Desgin Concept
- The whoe DAG is viewed as a big transformer, derived from `tfs_i`.
- Explicit CPU version or GPU version
- The net `convdag` is thin wrapper of the DAG, managing the training and testing

It is suggested that the following examples are read sequentially

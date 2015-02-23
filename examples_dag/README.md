# Examples for the use of DAG/GTN network 

Demonstrate how to build the DAG/GTN network by combining the Node Data 
(`n_data`) and the Transformers (`tf_xxx`) for your own task, i.e., how to 
wrap the APIs in the directory `./matlab_dag`. The GTN can be as simple as 
the common feed forward neural network, or as complicated as the TODO. The 
wrapping we demonstrate in this directory is similar to how
`./examples/cnn_train.m` and `./matlab/vl_simplenn.m` wrap the `vl_nnxxx` APIs. 

Recommended order to read:

1. `mnist_small_trOneBatch.m`: script showing how to train just one batch
2. `mnist_small_trBatches.m`: script showing how to train many batches
3. `mnist_small_tr.m`, `convdab_lenet.m`: script and wrapper class showing how to train with SGD 

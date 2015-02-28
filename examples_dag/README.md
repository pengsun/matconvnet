# Examples for the use of DAG/GTN network 

Demonstrate how to build the DAG/GTN network by combining the Node Data (`n_data.m`) and the Transformers (`tf_xxx`) for your own task, i.e., how to wrap the APIs in the directory `./matlab_dag`. The GTN can be as simple as 
the common feed forward neural network, or as complicated as the TODO. The wrapping we demonstrate in this directory is similar to how `./examples/cnn_train.m` and `./matlab/vl_simplenn.m` wrap the `vl_nnxxx` APIs. 

It is suggested that the following examples are read sequentially

## Training, CPU
1. `mnist_small_trOneBatch.m`: script showing how to train just one batch
2. `mnist_small_trBatches.m`: script showing how to train many batches
3. `mnist_small_tr_lenet.m`, `convdab_lenet.m`: script and wrapper class showing how to train with SGD 
4. `mnist_small_tr_lenetDropout.m`, `mnist_small_tr_lenetDropout.m`: yet another example. The network structure is almost the same with 3 except for an extra Dropout layer.
5. `mnist_small_tr_MLP.m`, `convdag_MLP.m`: example showing how to build MLP (Multi Layer Perceptron) by wrapping convnet
6. `mnist_small_tr_MLP2.m`, `convdag_MLP2.m`: examples for tfw_xxx. The network structure is the same with 5.
7. `mnist_small_tr_lenetMultiRes.m`, `convdag_lenetMultiRes.m`: examples showing the triangular connection at last layer, which is notably different with the feed forward net of list structure. See the comments in `convdag_lenetMultiRes.m` and the references therein.

## Training, GPU
1. `mnist_small_tr_gpu_lenet.m`, `convdab_gpu_lenet.m`: GPU versions of `mnist_small_tr_lenet.m` and  `convdab_lenet.m`

## Testing, CPU or GPU
After training, the learned model can be tested with the following scripts:

1. `mnist_small_te.m`: test and view the results for a single model
2. `mnist_small_te_all.m`: test and view the results for all the models, each being an epoch
3. `mnist_small_te_cmp.m`: test and compare the results for models from two directories, each being a network structure and parameter configuration

# Examples for the use of DAG/GTN network 

Demonstrate how to build the DAG/GTN network by combining the Node Data 
(`n_data`) and the Transformers (`tf_xxx`) for your own task. The GTN can be as 
simple as the common feed forward neural network, or as complicated as the 
TODO. Compared with how cnn_train.m and vl_simplenn.m wrap the `vl_nnxxx` 
APIs, the examples in this directory do the wrapping in an Object Oriented 
way, which should be hopefully more flexible and much easier to prototype 
your own idea.
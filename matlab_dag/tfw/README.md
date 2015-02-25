#TFW: transformer wrapper

## Design Concept
A transformer can be ssen as a DAG, or vice versa, a DAG can be seen as a
big transformer. In this directory, a couple of predefined transformers whose 
names startes with "tfw_" are provided. It would be more convenient for your 
own code to call those mature "layer-combination" that has been encapsulated 
as transformer wrappers. Examples include:
 * convolutional layer + pooling
 * linear layer + Relu + Dropout
 * TODO
Also, you can define your own "tfw" to make your code concise.

## How to
When customizing your own "tfw", just treat it as usual transformer by 
deriving from `tf_i.m`. Then do the following:
* mannually set the internal connection of the n_data() and other tf_xxx in the constructor
 ** note that the input and output of tfw should be linked to internal connection. See `tfw_LinReluDrop.m` for an example 
* override the fprop() and bprop(), if necessary
 ** the default behavior in tfw_i is to simply call the fprop() and bprop() for each internal transformer
That's it! However, please note that don't set the input (ob.i) and the 
output (ob.o) in the constructor -- leave them to caller's context.
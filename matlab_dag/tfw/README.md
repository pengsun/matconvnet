#TFW: transformer wrapper

## Design Concept
A transformer can be ssen as a DAG, or vice versa, a DAG can be seen as a
big transformer. In this directory, a couple of predefined transformers whose 
names start with "tfw_" are provided. It would be more convenient for your 
own code to call those mature "layer-combinations" that has been encapsulated 
as transformer wrappers. Examples include:
 * convolutional layer + pooling
 * linear layer + Relu + Dropout
 * TODO

Jus define your own "tfw_xxx" to make your code more concise.

## How to
When customizing your own "tfw", just treat it as usual transformer by deriving from `tf_i.m`. Then do the following:
- Mannually set the internal connection of the n_data() and other tf_xxx in the constructor
- Override the fprop() and bprop(). Copy the outer and inner n_data() properly before calling fprop() and bprop(). See `tfw_LinReluDrop.m` for an example

That's it! 

# Wrapper for Directed Acyclic Graph (DAG), 

DAG is a.k.a. Graph Transformer Network (GTN). Examples include check 
reader [], secen parser [], pose estimater [], multi resolution CNN [2], 
Cascade Neural Network [3], tunable Cascade Pose Estimate [4], etc. The
feed-forward net in bi-directional list is a special case of DAG/GTN.

## Design Concept
Graph as neural network; Node as data, including hidden variables, 
instances and labels (source/root), parameters (source/root), loss 
(sink/leaf), etc; Layer/Transformer as edge, including convolutional layer, 
pooling layer, loss layer, etc. 

The wrapping is in an Object Oriented way, which should be hopefully more 
flexible and much easier to prototype your own idea by incorporating 
customized components (i.e., Node Data or Transformers). The Abstraction 
Penalty (overhead for the wrapping data structure) should be negligible.

## Reference
[]
[]
[]
[1] LeCun, Gradient based Learning.
[2] Xiaoou Tang's paper?
[3] Matlab Neural Network Toolbox
[4] ...
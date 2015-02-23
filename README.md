# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images. Please visit
the [homepage](http://www.vlfeat.org/matconvnet) to know more.


---------------------
Forked by Peng Sun and modified for customized use. Main purposes:
- Directed Acyclic Graph (**DAG**), a.k.a. Graph Transformer Network (**GTN**), see `./matlab_dag/README.md`
- Recurrent Network, which is no more than deep structure with shared 
parameters across layers when unfolding
- Vector-Valued Regression (e.g., face pose estimate)

## Install
1. Follow the original instructions for setup, which would compile the mex 
code, add to path the directory `./matlab`.
2. Add directory `./matlab_dag` to path by running in command window the 
following code:
``` matlab
dag_setup;
```
or doing this mannually (e.g., File menu -> Set Path)

When it is done, run the m files in directory `examples_dag` for examples.

## TODO
 - DAG/GTN implementations (wrappers)
   - parametric transformer 
     - [x] convolution
   - non-parametric transformer
     - [x] pooling
     - [ ] dropout
     - [x] relu
     - [ ] lateral normalization 
   - non-parametric auxiliary transformer
     - [ ] multiplex/add
     - [ ] split/concatenate
   - loss transformer 
     - [x] LSE (Least Square Error)
     - [ ] Logit (softmax) 
   - wrapper/example code in `examples_dag`
     - [x] basic training
     - [ ] training with validation
     - [ ] A simple DAG other than the pure feed forward structure
   - Misceallaneous
     - [ ] GPU version
 - Extension of `vl_simplenn.m` and associated files
   - [x] Least Square Loss
   - [x] Code for direct CNN Testing

## FIXME
 - [ ] Problematic when batchSize = 1 ?

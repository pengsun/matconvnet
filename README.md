# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images. Please visit
the [homepage](http://www.vlfeat.org/matconvnet) to know more.


---------------------
Forked by Peng Sun and modified for personal use. Main purposes:
- Directed Acyclic Graph (DAG), a.k.a. Graph Transformer Network (GTN), see
  README.md in `./matlab_dag`
- Vector-Valued Regression (e.g., face pose estimate)

## Install
1. Follow the original instructions for setup, which compiles the mex code,
   add to path the directory `./matlab`.
2. Add the `./matlab_dag` directory to path by running
``` matlab
dag_path.add();
```
3. Done. See the directory `examples_dag` for examples.

## TODO
 - DAG/GTN implementations (wrappers)
   - parametric transformer 
     - [ ] convolution
   - non-parametric transformer (pool, relu)
     - [ ] pooling
     - [ ] dropout
     - [ ] relu
     - [ ] normalization 
   - loss transformer 
     - [ ] LSE (Least Square Error)
     - [ ] Logit (softmax) 
 - Extension of `vl_simplenn.m` and associated files
   - [x] ~~Least Square Loss~~
   - [x] ~~Code for direct CNN Testing~~

## FIXME
 - [ ] Problematic when batchSize = 1 ?

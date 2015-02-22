# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images. Please visit
the [homepage](http://www.vlfeat.org/matconvnet) to know more.


---------------------
Forked by Peng Sun and modified for personal use. Main purposes:
- Directed Acyclic Graph (DAG), a.k.a. Graph Transformer Network (GTN), see
  README.md in matlab_dag
- Vector-Valued Regression (e.g., face pose estimate)

## TODO
 - DAG/GTN implementation (wrapper)
   - parametric transformer (conv, pool)
   - non-parametric transformer (pool)
   - loss transformer (lse) 
 - ~~Least Square Loss~~
 - ~~Code for direct CNN Testing~~

## FIXME
 * Problematic when batchSize = 1 ?

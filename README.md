# Code for the paper: Embedding-based query expansion for weighted sequential dependence retrieval model

PDF: [paper.pdf](https://github.com/teanalab/SWDM/blob/master/paper.pdf)

## Main Dependencies:
+ Python 3.6
+ Indri 5.11
+ [pyndri](https://github.com/teanalab/pyndri)
+ Gensim 1.0.1

## Running the Code
You may call queryWeightsOptimizer to learn the weights of the proposed
method. This code uses a cross-validation approach to obtain the results.
```bash
./queries/queryWeightsOptimizer.py
```

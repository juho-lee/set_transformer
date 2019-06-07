# set_transformer
Official PyTorch implementation of the paper 
[Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks
](http://proceedings.mlr.press/v97/lee19d.html).

## Requirements
- Python 3
- torch >= 1.0
- matplotlib
- scipy
- tqdm

## Abstract

Many machine learning tasks such as multiple instance learning, 3D shape recognition, and few-shot image classification are defined on sets of instances. 
Since solutions to such problems do not depend on the order of elements of the set, models used to address them should be permutation invariant. 
We present an attention-based neural network module, the Set Transformer, specifically designed to model interactions among elements in the input set.
The model consists of an encoder and a decoder, both of which rely on attention mechanisms.
In an effort to reduce computational complexity, we introduce an attention scheme inspired by inducing point methods from sparse Gaussian process literature.
It reduces the computation time of self-attention from quadratic to linear in the number of elements in the set.
 We show that our model is theoretically attractive and we evaluate it on a range of tasks, demonstrating the state-of-the-art performance compared to recent methods for set-structured data.
 
## Contents

This repository implements the maximum value regression (section 5.1) and the amortized clustering (section 5.3) 
experiments in the [paper](http://proceedings.mlr.press/v97/lee19d.html).
The maximum value regression experiment is reproduced in `max_regression_demo.ipynb`.

To run amortized clustering experiment with Set Transformer, run
```
python run.py --net=set_transformer
```
To run the same experiment with Deep Sets, run
```
python run.py --net=deepset
```
 
## Reference

If you found the provided code useful, please consider citing our work.

```
@InProceedings{lee2019set,
    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    pages={3744--3753},
    year={2019}
}
```

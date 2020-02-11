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

## Experiments

This repository implements the
maximum value regression (section 5.1),
amortized clustering (section 5.3),
and point cloud classification (section 5.5)
experiments in the [paper](http://proceedings.mlr.press/v97/lee19d.html).

### Maximum Value Regression

This experiment is reproduced in `max_regression_demo.ipynb`.

### Amortized Clustering

To run the amortized clustering experiment with Set Transformer, run
```
python run.py --net=set_transformer
```
To run the same experiment with Deep Sets, run
```
python run.py --net=deepset
```

### Point Cloud Classification
We used the same preprocessed ModelNet40 dataset used in the [DeepSets paper](https://papers.nips.cc/paper/6931-deep-sets).
We cannot publicly share this file due to copyright and license issues.
To run this code, you must obtain the preprocessed dataset "ModelNet40_cloud.h5".
We recommend using multiple GPUs for this experiment; we used 8 Tesla P40s.

To run the point cloud classification experiment, run
```
python main_pointcloud.py --batch_size 256 --num_pts 100
python main_pointcloud.py --batch_size 256 --num_pts 1000
python main_pointcloud.py --batch_size 256 --num_pts 5000
```

The hyperparameters here were minimally tuned yet reproduced the results in the paper.
It is likely that further tuning will get better results.

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

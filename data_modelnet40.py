import numpy as np
import h5py


def rotate_z(theta, x):
    theta = np.expand_dims(theta, 1)
    outz = np.expand_dims(x[:, :, 2], 2)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xx = np.expand_dims(x[:, :, 0], 2)
    yy = np.expand_dims(x[:, :, 1], 2)
    outx = cos_t * xx - sin_t * yy
    outy = sin_t * xx + cos_t * yy
    return np.concatenate([outx, outy, outz], axis=2)


def augment(x):
    bs = x.shape[0]
    # rotation
    min_rot, max_rot = -0.1, 0.1
    thetas = np.random.uniform(min_rot, max_rot, [bs, 1]) * np.pi
    rotated = rotate_z(thetas, x)
    # scaling
    min_scale, max_scale = 0.8, 1.25
    scale = np.random.rand(bs, 1, 3) * (max_scale - min_scale) + min_scale
    return rotated * scale


def standardize(x):
    clipper = np.mean(np.abs(x), (1, 2), keepdims=True)
    z = np.clip(x, -100 * clipper, 100 * clipper)
    mean = np.mean(z, (1, 2), keepdims=True)
    std = np.std(z, (1, 2), keepdims=True)
    return (z - mean) / std


class ModelFetcher(object):
    def __init__(self, fname, batch_size, down_sample=10, do_standardize=True, do_augmentation=False):

        self.fname = fname
        self.batch_size = batch_size
        self.down_sample = down_sample

        with h5py.File(fname, 'r') as f:
            self._train_data = np.array(f['tr_cloud'])
            self._train_label = np.array(f['tr_labels'])
            self._test_data = np.array(f['test_cloud'])
            self._test_label = np.array(f['test_labels'])

        self.num_classes = np.max(self._train_label) + 1

        self.num_train_batches = len(self._train_data) // self.batch_size
        self.num_test_batches = len(self._test_data) // self.batch_size

        self.prep1 = standardize if do_standardize else lambda x: x
        self.prep2 = (lambda x: augment(self.prep1(x))) if do_augmentation else self.prep1

        assert len(self._train_data) > self.batch_size, \
            'Batch size larger than number of training examples'

        # select the subset of points to use throughout beforehand
        self.perm = np.random.permutation(self._train_data.shape[1])[::self.down_sample]

    def train_data(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self._train_data)
        np.random.set_state(rng_state)
        np.random.shuffle(self._train_label)
        return self.next_train_batch()

    def next_train_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._train_data)
        perm = self.perm
        batch_card = len(perm) * np.ones(self.batch_size, dtype=np.int32)
        while end < N:
            yield self.prep2(self._train_data[start:end, perm]), batch_card, self._train_label[start:end]
            start = end
            end += self.batch_size

    def test_data(self):
        return self.next_test_batch()

    def next_test_batch(self):
        start = 0
        end = self.batch_size
        N = len(self._test_data)
        batch_card = (self._train_data.shape[1] // self.down_sample) * np.ones(self.batch_size, dtype=np.int32)
        while end < N:
            yield self.prep1(self._test_data[start:end, 1::self.down_sample]), batch_card, self._test_label[start:end]
            start = end
            end += self.batch_size

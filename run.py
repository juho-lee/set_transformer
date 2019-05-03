import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import SetTransformer, DeepSet
from mixture_of_mvns import MixtureOfMVNs
from mvn_diag import MultivariateNormalDiag

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--num_bench', type=int, default=100)
parser.add_argument('--net', type=str, default='set_transformer')
parser.add_argument('--B', type=int, default=10)
parser.add_argument('--N_min', type=int, default=300)
parser.add_argument('--N_max', type=int, default=600)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--num_steps', type=int, default=50000)
parser.add_argument('--test_freq', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=400)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

B = args.B
N_min = args.N_min
N_max = args.N_max
K = args.K

D = 2
mvn = MultivariateNormalDiag(D)
mog = MixtureOfMVNs(mvn)
dim_output = 2*D

if args.net == 'set_transformer':
    net = SetTransformer(D, K, dim_output).cuda()
elif args.net == 'deepset':
    net = DeepSet(D, K, dim_output).cuda()
else:
    raise ValueError('Invalid net {}'.format(args.net))
benchfile = os.path.join('benchmark', 'mog_{:d}.pkl'.format(K))

def generate_benchmark():
    if not os.path.isdir('benchmark'):
        os.makedirs('benchmark')
    N_list = np.random.randint(N_min, N_max, args.num_bench)
    data = []
    ll = 0.
    for N in tqdm(N_list):
        X, labels, pi, params = mog.sample(B, N, K, return_gt=True)
        ll += mog.log_prob(X, pi, params).item()
        data.append(X)
    bench = [data, ll/args.num_bench]
    torch.save(bench, benchfile)

save_dir = os.path.join('results', args.net, args.run_name)
def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if not os.path.isfile(benchfile):
        generate_benchmark()

    bench = torch.load(benchfile)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(args.run_name)
    logger.addHandler(logging.FileHandler(
        os.path.join(save_dir,
            'train_'+time.strftime('%Y%m%d-%H%M')+'.log'),
        mode='w'))
    logger.info(str(args) + '\n')

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    tick = time.time()
    for t in range(1, args.num_steps+1):
        if t == int(0.5*args.num_steps):
            optimizer.param_groups[0]['lr'] *= 0.1
        net.train()
        optimizer.zero_grad()
        N = np.random.randint(N_min, N_max)
        X = mog.sample(B, N, K)
        ll = mog.log_prob(X, *mvn.parse(net(X)))
        loss = -ll
        loss.backward()
        optimizer.step()

        if t % args.test_freq == 0:
            line = 'step {}, lr {:.3e}, '.format(
                    t, optimizer.param_groups[0]['lr'])
            line += test(bench, verbose=False)
            line += ' ({:.3f} secs)'.format(time.time()-tick)
            tick = time.time()
            logger.info(line)

        if t % args.save_freq == 0:
            torch.save({'state_dict':net.state_dict()},
                    os.path.join(save_dir, 'model.tar'))

    torch.save({'state_dict':net.state_dict()},
        os.path.join(save_dir, 'model.tar'))

def test(bench, verbose=True):
    net.eval()
    data, oracle_ll = bench
    avg_ll = 0.
    for X in data:
        X = X.cuda()
        avg_ll += mog.log_prob(X, *mvn.parse(net(X))).item()
    avg_ll /= len(data)
    line = 'test ll {:.4f} (oracle {:.4f})'.format(avg_ll, oracle_ll)
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(args.run_name)
        logger.addHandler(logging.FileHandler(
            os.path.join(save_dir, 'test.log'), mode='w'))
        logger.info(line)
    return line

def plot():
    net.eval()
    X = mog.sample(B, np.random.randint(N_min, N_max), K)
    pi, params = mvn.parse(net(X))
    ll, labels = mog.log_prob(X, pi, params, return_labels=True)
    fig, axes = plt.subplots(2, B//2, figsize=(7*B//5,5))
    mog.plot(X, labels, params, axes)
    plt.show()

if __name__ == '__main__':
    if args.mode == 'bench':
        generate_benchmark()
    elif args.mode == 'train':
        train()
    elif args.mode == 'test':
        bench = torch.load(benchfile)
        ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
        net.load_state_dict(ckpt['state_dict'])
        test(bench)
    elif args.mode == 'plot':
        ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
        net.load_state_dict(ckpt['state_dict'])
        plot()

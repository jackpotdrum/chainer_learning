import chainer
import chainer.links as L
import chainer.functions as F
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from chainer.datasets import TupleDataset, split_dataset_random
from chainer.iterators import SerialIterator
from chainer import optimizers
from chainer.optimizer_hooks import WeightDecay
from chainer.serializers import save_npz


class Net(chainer.Chain):
    def __init__(self, n_in=4, n_hidden=3, n_out=3):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.l3 = L.Linear(n_hidden, n_out)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        return h

x, t = load_iris(return_X_y=True)
x = x.astype('float32')
t = t.astype('int32')

dataset = TupleDataset(x, t)

train_val, test = split_dataset_random(dataset, int(len(dataset) * 0.7), seed=0)
train, valid = split_dataset_random(dataset, int(len(train_val) * 0.7), seed=0)

train_iter = SerialIterator(train, batch_size=4, repeat=True, shuffle=True)
minibatch = train_iter.next()

net = Net()

optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
optimizer.setup(net)

for param in net.params():
    if param.name != 'b':
        param.update_rule.add_hook(WeightDecay(0.0001))

gpu_id = 0
n_batch = 64
n_epoch = 50

net.to_gpu(gpu_id)

results_train, results_valid = {}, {}
results_train['loss'], results_train['accuracy'] = [], []
results_valid['loss'], results_valid['accuracy'] = [], []

train_iter.reset()

count = 1

for epoch in range(n_epoch):
    while True:
        train_batch = train_iter.next()

        x_train, t_train = chainer.dataset.concat_examples(train_batch, gpu_id)

        y_train = net(x_train)
        loss_train = F.softmax_cross_entropy(y_train, t_train)
        acc_train = F.accuracy(y_train, t_train)

        net.cleargrads()
        loss_train.backward()

        optimizer.update()

        count += 1

        if train_iter.is_new_epoch:
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x_valid, t_valid = chainer.dataset.concat_examples(valid, gpu_id)
                y_valid = net(x_valid)
                loss_valid = F.softmax_cross_entropy(y_valid, t_valid)
                acc_valid = F.accuracy(y_valid, t_valid)

            loss_train.to_cpu()
            loss_valid.to_cpu()
            acc_train.to_cpu()
            acc_valid.to_cpu()

            print(
                'epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}, '
                 'acc (train): {:.4f}, acc (valid): {:.4f}'.format(epoch, count, 
                     loss_train.array.mean(), loss_valid.array.mean(),
                     acc_train.array.mean(), acc_valid.array.mean())
                 )
            
            results_train['loss'].append(loss_train.array)
            results_train['accuracy'].append(acc_train.array)
            results_valid['loss'].append(loss_valid.array)
            results_valid['accuracy'].append(acc_valid.array)

            break
  
plt.plot(results_train['loss'], label='train')
plt.plot(results_valid['loss'], label='valid')
plt.legend()
plt.show()

plt.plot(results_train['accuracy'], label='train')
plt.plot(results_valid['accuracy'], label='valid')
plt.legend()
plt.show()

x_test, t_test = chainer.dataset.concat_examples(test, device=gpu_id)
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = net(x_test)
    loss_test = F.softmax_cross_entropy(y_test, t_test)
    acc_test = F.accuracy(y_test, t_test)
print('test loss: {:.4f}'.format(loss_test.array.get()))
print('test accuracy: {:.4f}'.format(acc_test.array.get()))

net.to_cpu()
save_npz('net_npz', net)

params = np.load('net_npz')
for key, param in params.items():
    print(key, ':\t', param.shape)
import chainer

from sklearn.datasets import load_iris
from chainer.datasets import TupleDataset, split_dataset_random
from chainer.iterators import SerialIterator
from chainer.links as L
from chainer.functions as F
from chainer import optimizers
from chainer.optimizer_hooks import WeightDecay


x, t = load_iris(return_X_y=True)
x = x.astype('float32')
t = t.astype('int32')

dataset = TupleDataset(x, t)

train_val, test = split_dataset_random(dataset, int(len(dataset) * 0.7), seed=0)
train, valid = split_dataset_random(dataset, int(len(train_val) * 0.7), seed=0)

train_iter = SerialIterator(train, batch_size=4, repeat=True, shuffle=True)
minibatch = train_iter.next()

class net(chainer.Chain):
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

net = Net()

optimizer = optimizers.SGD(lr=0.001)
optimizer.setup(net)

for param in net.params():
    if param.name != 'b':
        param.update_rule.add_hook(WeightDecay(0.0001))
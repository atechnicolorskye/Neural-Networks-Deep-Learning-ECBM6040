"""
Source Code for Homework 4.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

"""

from collections import OrderedDict
import numpy
import os
import random

import theano
from theano import tensor as T

from hw4_utils import shared_dataset
from hw4_nn import myMLP, train_nn, LogisticRegression

# theano.config.exception_verbosity='high'

def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(num, nbit))
    Y = numpy.mod(numpy.sum(X, axis=1), 2)
    # X = 2 * X - 1

    print X.shape
    return X, Y

def gen_parity_pair_rnn(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(num, nbit))
    Y = numpy.array(X)
    for i in range(1,nbit):
        Y[:,i] += Y[:, i-1]

    Y = numpy.mod(Y, 2)
    return X, Y

#TODO: implement RNN class to learn parity function
class RNN(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, input, nh, nbits, batch_size):
        """Initialize the parameters for the RNN

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nbits: int
        :param nbits: number of bits in parity function

        :type batch_size: int
        :param batch_size:
        """
        # parameters of the model
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        # self.w = theano.shared(name='w',
        #                        value=0.2 * numpy.random.uniform(-1.0, 1.0,
        #                        (nh, 2))
        #                        .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        # self.b = theano.shared(name='b',
        #                        value=numpy.zeros(2,
        #                        dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        # http://deeplearning.net/software/theano/library/scan.html
        # From Simple accumulation into a scalar, ditching lambda onwards

        # Enable broadcasting to keep dimensions, failed as non-differentiable
        # self.wx = T.addbroadcast(self.wx, 0)

        def recurrence(i, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(input[:, i].dimshuffle(0, 'x'), self.wx)
                                 + T.dot(h_tm1, self.wh) + self.bh)
            return h_t

        # self.h0 = T.addbroadcast(self.h0, 0)
        h0 = T.zeros(shape=(input.shape[0], nh)) + self.h0

        h, _ = theano.scan(fn=recurrence,
                           sequences=T.arange(input.shape[1]),
                           outputs_info=h0)
        # Can't use numpy.zeros((batch_size,nh)) as it doesn't scale with different batch sizes.

        self.logRegressionLayer = LogisticRegression(
            input=h[-1],
            n_in=nh,
            n_out=2
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = [self.wx, self.wh, self.bh, self.h0] + self.logRegressionLayer.params

        self.input = input

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))

class RNN_ALL_Y(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, input_x, nh):
        """Initialize the parameters for the RNN

        :type nh: int
        :param nh: dimension of the hidden layer
        """
        # parameters of the model
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx)
                                 + T.dot(h_tm1, self.wh) + self.bh)
            return h_t

        h, _ = theano.scan(fn=recurrence,
                              sequences=input_x,
                              outputs_info=[self.h0],
                              n_steps=input_x.shape[0])

        self.logRegressionLayer = LogisticRegression(
            input=h,
            n_in=nh,
            n_out=2
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = [self.wx, self.wh, self.bh] + self.logRegressionLayer.params

        self.input = input_x

#TODO: build and train a MLP to learn parity function
def test_mlp_parity(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
                    batch_size=20, n_hidden=300, n_hiddenLayers=1, nbit=8, verbose=False):
    print test_mlp_parity.__name__, n_epochs, n_hidden, n_hiddenLayers, nbit

    # Generate datasets
    train_set = gen_parity_pair(nbit, 1000)
    valid_set = gen_parity_pair(nbit, 500)
    test_set  = gen_parity_pair(nbit, 100)

    # Convert raw dataset to Theano shared variablesself.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    # Compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // 100
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // 100

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as a matrix
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # TODO: construct a neural network, either MLP or CNN.
    classifier = myMLP(
       rng=rng,
       input=x,
       n_in=nbit,
       n_hidden=n_hidden,
       n_out=2,
       n_hiddenLayers=n_hiddenLayers
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * 100:(index + 1) * 100],
            y: test_set_y[index * 100:(index + 1) * 100]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * 100:(index + 1) * 100],
            y: valid_set_y[index * 100:(index + 1) * 100]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

#TODO: build and train a RNN to learn parity function
def test_rnn_parity(**kwargs):
    """
    Wrapper function for training and testing RNNSLU

    :type lr: float
    :param lr: learning rate used (factor for the stochastic gradient

    :type nhidden: int
    :param n_hidden: number of hidden units

    :type nbits: int
    :param nbits: number of bits in parity function

    :type n_batch: int
    :param n_batch: batch_size

    :type nepochs: int
    :param nepochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """
    param = {
        'lr': 0.75,
        'verbose': True,
        'nhidden': 20,
        'nbit': 12,
        'seed': 345,
        'n_batch': 1000,
        'nepochs': 1000,
        'folder': '../result'}
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    # Generate datasets
    train_set = gen_parity_pair(param['nbit'], 1000)
    valid_set = gen_parity_pair(param['nbit'], 500)
    test_set  = gen_parity_pair(param['nbit'], 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    # Compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // param['n_batch']
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // 100
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // 100

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as a matrix
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rnn = RNN(
        input=x,
        nh=param['nhidden'],
        nbits=param['nbit'],
        batch_size=param['n_batch'])

    rnn.load(param['folder'])

    # train with early stopping on validation set
    print('... training')

    cost = (
        rnn.negative_log_likelihood(y)
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=rnn.errors(y),
        givens={
            x: test_set_x[index * 100:(index + 1) * 100],
            y: test_set_y[index * 100:(index + 1) * 100]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=rnn.errors(y),
        givens={
            x: valid_set_x[index * 100:(index + 1) * 100],
            y: valid_set_y[index * 100:(index + 1) * 100]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, p) for p in rnn.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (p, p - param['lr'] * gparam)
        for p, gparam in zip(rnn.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * param['n_batch']:(index + 1) * param['n_batch']],
            y: train_set_y[index * param['n_batch']:(index + 1) * param['n_batch']]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, param['nepochs'], param['verbose'])

    rnn.save(param['folder'])

def test_rnn_parity_all_y(**kwargs):
    """
    Wrapper function for training and testing RNNSLU

    :type lr: float
    :param lr: learning rate used (factor for the stochastic gradient

    :type nhidden: int
    :param n_hidden: number of hidden units

    :type nbits: int
    :param nbits: number of bits in parity function

    :type nepochs: int
    :param nepochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """
    param = {
        'lr': 0.05,
        'verbose': True,
        'nhidden': 12,
        'nbit': 12,
        'seed': 345,
        'nepochs': 400}
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    # Generate datasets
    train_set = gen_parity_pair_rnn(param['nbit'], 1000)
    valid_set = gen_parity_pair_rnn(param['nbit'], 500)
    test_set  = gen_parity_pair_rnn(param['nbit'], 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    # Compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as a matrix
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rnn = RNN_ALL_Y(
        input_x=x,
        nh=param['nhidden'])

    # train with early stopping on validation set
    print('... training')

    cost = (
        rnn.negative_log_likelihood(y)
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=rnn.errors(y),
        givens={
            # Shuffle dim as dot product has been changed
            x: test_set_x[index * 1:(index + 1) * 1].dimshuffle(1,0),
            y: test_set_y[index * 1:(index + 1) * 1][0]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=rnn.errors(y),
        givens={
            x: valid_set_x[index * 1:(index + 1) * 1].dimshuffle(1,0),

            y: valid_set_y[index * 1:(index + 1) * 1][0]

        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, p) for p in rnn.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (p, p - param['lr'] * gparam)
        for p, gparam in zip(rnn.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * 1: (index + 1) * 1].dimshuffle(1,0),
            y: train_set_y[index * 1: (index + 1) * 1][0]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, param['nepochs'], param['verbose'])


if __name__ == '__main__':
    test_rnn_parity()
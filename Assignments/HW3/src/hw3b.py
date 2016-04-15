"""
Source Code for Homework 3.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import cv2
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn

# Build Gabor filters
def build_gabor(ksize, num, lmbda):
    filters = []
    ksize = ksize
    for theta in numpy.arange(0, numpy.pi, numpy.pi / num):
        kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lmbda, 0.5, 0,  ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

# TODO
def test_lenet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512],
            batch_size=200, filter_size=5, dnn_layers=1, n_hidden=500, gabor=False, lmbda=None, verbose=False):
    """
    Wrapper function for testing LeNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """
    print test_lenet.__name__, nkerns, filter_size, gabor, lmbda

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    if gabor is True:
        # Generate Gabor filters
        filters = build_gabor(filter_size, nkerns[0], lmbda)
        # filters = numpy.array([filters[i][0] for i in range(len(filters))])
        filters = numpy.array([filters[i] for i in range(len(filters))])
        # print filters.shape
        filter_weights = numpy.tile(filters, (1, 3, 1)).reshape(nkerns[0], 3, filter_size, filter_size)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 3, 32, 32),
            filter_shape=(nkerns[0], 3, filter_size, filter_size),
            poolsize=(2,2),
            weights = filter_weights
        )
        print 'gabor filter weights are working'
    else:
        # TODO: Construct the first convolutional pooling layer
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 3, 32, 32),
            filter_shape=(nkerns[0], 3, filter_size, filter_size),
            poolsize=(2,2)
        )

    # TODO: Construct the second convolutional pooling layer
    i_s_1 = (32 - filter_size + 1) / 2

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], i_s_1, i_s_1),
        filter_shape=(nkerns[1], nkerns[0], filter_size, filter_size),
        poolsize=(2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)

    # TODO: construct a fully-connected sigmoidal layer
    i_s_2 = (i_s_1 - filter_size + 1) / 2

    if hasattr(n_hidden, '__iter__'):
        assert(len(n_hidden) == dnn_layers)
    else:
        n_hidden = (n_hidden,)*dnn_layers

    DNN_Layers = []
    for i in xrange(dnn_layers):
        h_input = layer2_input if i == 0 else DNN_Layers[i-1].output
        h_in = nkerns[1] * i_s_2 * i_s_2 if i == 0 else n_hidden[i-1]
        DNN_Layers.append(
            HiddenLayer(
                rng=rng,
                input=h_input,
                n_in=h_in,
                n_out=n_hidden[i],
                activation=T.tanh
        ))

    # layer2 = HiddenLayer(
    #     rng,
    #     input=layer2_input,
    #     n_in=nkerns[1] * i_s_2 * i_s_2,
    #     n_out=500,
    #     activation=T.tanh
    # )

    # TODO: classify the values of the fully-connected sigmoidal layer
    LR_Layer = LogisticRegression(
        input=DNN_Layers[-1].output,
        n_in=n_hidden[i],
        n_out=10
    )

    # the cost we minimize during training is the NLL of the model
    cost = LR_Layer.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        LR_Layer.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        LR_Layer.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = LR_Layer.params
    for layer in DNN_Layers:
        params += layer.params
    if gabor is True:
        print 'gabor params is workings'
        params += layer1.params
    else:
        params += layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
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

    # Uncomment for bonus part 1
    # return layer0.W.get_value()

# TODO
def test_convnet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512, 20],
        batch_size=200, filter_size=5, verbose=False):
    """
    Wrapper function for testing Multi-Stage ConvNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """
    print test_convnet.__name__, nkerns, filter_size

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # TODO: Construct the first convolutional pooling layer:
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, filter_size, filter_size),
        poolsize=(2, 2)
    )

    # TODO: Construct the second convolutional pooling layer
    # Calculate image size post-pooling

    i_s_1 = (32 - filter_size + 1) / 2

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], i_s_1, i_s_1),
        filter_shape=(nkerns[1], nkerns[0], filter_size, filter_size),
        poolsize=(2, 2)
    )

    # Combine Layer 0 output and Layer 1 output
    # TODO: downsample the first layer output to match the size of the second
    # layer output.

    # Calculate image size post-pooling
    i_s_2 = (i_s_1 - filter_size + 1) / 2

    d_s = int(numpy.ceil(float(i_s_1) / i_s_2))

    if i_s_1 / d_s == i_s_2:
        layer0_output_ds = downsample.max_pool_2d(
                input=layer0.output,
                ds=(d_s, d_s), # TDOD: change ds
                ignore_border=False
        )
    else:
        layer0_output_ds = downsample.max_pool_2d(
                input=layer0.output,
                ds=(d_s - 1, d_s - 1), # TDOD: change ds
                ignore_border=False
        )

    layer0_output_ds = layer0_output_ds[:, :, :i_s_2, :i_s_2]

    # concatenate layer
    layer2_input = T.concatenate([layer1.output, layer0_output_ds], axis=1)

    # TODO: Construct the third convolutional pooling layer
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer2_input,
        image_shape=(batch_size, nkerns[0] + nkerns[1], i_s_2, i_s_2),
        filter_shape=(nkerns[2], nkerns[0] + nkerns[1], i_s_2, i_s_2),
        poolsize=(1, 1)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 1 * 1).
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 1 * 1,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output,
        n_in=500,
        n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
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
# TODO
def test_CDNN(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512],
            batch_size=200, filter_size=5, dnn_layers=2, n_hidden=500, verbose=False):
    """
    Wrapper function for testing CNN in cascade with DNN
    """
    test_lenet(learning_rate=learning_rate, n_epochs=n_epochs, nkerns=nkerns,
            batch_size=batch_size, filter_size=filter_size, dnn_layers=dnn_layers,
            n_hidden=n_hidden, gabor=False, verbose=verbose)


if __name__ == "__main__":
    # test_lenet(nkerns=[8, 256], filter_size=7, verbose=True)
    for i in [2, 2.1, 3, 4, 5]:
    # print 'Test'
        test_lenet(nkerns=[16, 512], filter_size=5, gabor=True, lmbda=i, verbose=True)
    # test_lenet(nkerns=[32, 1024], filter_size=7, verbose=True)
    # test_lenet(nkerns=[16, 512], filter_size=3, verbose=True)
    # test_convnet(learning_rate=0.1, n_epochs=1000, nkerns=[8, 256, 20],
    #     batch_size=200, filter_size=3, verbose=False)
    # test_convnet(learning_rate=0.1, n_epochs=1000, nkerns=[8, 256, 20],
    #     batch_size=200, filter_size=5, verbose=False)
    # test_convnet(learning_rate=0.1, n_epochs=1000, nkerns=[32, 1024, 20],
    #     batch_size=200, filter_size=3, verbose=False)
    # test_convnet(learning_rate=0.1, n_epochs=1000, nkerns=[32, 1024, 20],
    #     batch_size=200, filter_size=5, verbose=False)
    # test_convnet(verbose=True)
    # test_lenet(nkerns=[8, 256], filter_size=3, verbose=False)
    # test_lenet(nkerns=[32, 1024], filter_size=3, verbose=False)
    # test_lenet(nkerns=[32, 1024], filter_size=5, verbose=False)
    # test_CDNN()

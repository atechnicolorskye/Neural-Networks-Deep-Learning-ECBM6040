import numpy as np
import cPickle as pickle
import os
import gzip
import tarfile
import theano
import theano.tensor as T
from sklearn.cross_validation import train_test_split

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_cifar(d=10, borrow = True):
    '''
    Load the ATIS dataset

    :type foldnum: int
    :param foldnum: fold number of the ATIS dataset, ranging from 0 to 4.

    '''

    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    cifar100_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    url=cifar10_url
    if d==100:
        url = cifar100_url
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            print('Downloading data from %s' % url)
            urllib.request.urlretrieve(url, new_path)
        f_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data"
        )
        tar = tarfile.open(new_path)
        tar.extractall(f_path)
        tar.close()
        final_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            "cifar-10-batches-py"
        )
        return final_path


    filename = check_dataset('cifar-10-python.tar.gz')
    train_data = [None]*5
    l = 0
    for i in xrange(5):
        data_path = filename+'/data_batch_'+str(i+1)
        train_data[i] = unpickle(data_path)
        l+=len(train_data[i]['labels'])
    test_data = unpickle(filename+'/test_batch')
    label_names = unpickle(filename+'/batches.meta')


    train_img = np.zeros((l,3,32,32))
    train_labels = np.array(())
    j=0
    for i in xrange(5):
        imgs = train_data[i]['data']
        labels = train_data[i]['labels']
        train_labels = np.append(train_labels,labels)
        for img in imgs:
            train_img[j] = img.reshape(3,32,32)
            j+=1
    j = 0
    test_img = np.zeros((len(test_data['data']),3,32,32))
    test_labels = test_data['labels']
    for img in test_data['data']:
        test_img[j]=img.reshape(3,32,32)
        j+=1

    train_img /= 255.0
    test_img /= 255.0

    train_img, valid_img, train_labels, valid_labels = train_test_split(train_img, train_labels, test_size=0.1)

    train_img = theano.shared(np.asarray(train_img,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    test_img = theano.shared(np.asarray(test_img,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    valid_img = theano.shared(np.asarray(valid_img,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    valid_labels = theano.shared(np.asarray(valid_labels,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    train_labels = theano.shared(np.asarray(train_labels,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    test_labels = theano.shared(np.asarray(test_labels,
                                           dtype=theano.config.floatX),
                             borrow=borrow)

    return [train_img,T.cast(train_labels,'int32'),valid_img,T.cast(valid_labels,'int32'),test_img,T.cast(test_labels,'int32')]




if __name__ == '__main__':
    load_cifar(10)
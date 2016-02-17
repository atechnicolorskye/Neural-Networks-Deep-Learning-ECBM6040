import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from os import chdir, getcwd, walk
from PIL import Image


'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''


def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plt.subplot(3, 3, i * 3 + j + 1)
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('output/hw1b_im{0}.png'.format(im_num))
    plt.close(f)
    # plt.show()


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of a image

    imname: string
        name of file where image will be saved.
    '''

    '''
    Stitch = []

    for i in range(0, 4):
        Stitch.append(D.T[i * 4].reshape(sz, sz))
        # print i * 4
        for j in range(1, 4):
            # print i * 4, i * 4 + j
            Stitch[i] = np.hstack((Stitch[i], D.T[i * 4 + j].reshape(sz, sz)))

    for i in range(1, len(Stitch)):
        Stitch[0] = np.vstack((Stitch[0], Stitch[i]))

    plt.imshow(Stitch[0], cmap=plt.gray())
    plt.savefig(imname)
    '''
    f, axarr = plt.subplots(4, 4)

    # print D.shape

    for i in range(4):
        for j in range(4):
            # plt.subplot(4, 4, i * 4 + j + 1)
            print i * 4 + j + 1
            plt.imshow(D.T[i * 4 + j].reshape(sz, sz), cmap=plt.gray())

    # plt.show()
    plt.savefig(imname)


def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction

        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image

        ax: the axis on which the image will be plotted
    '''
    # raise NotImplementedError
    # print D.shape, c.shape
    H, W = X_mn.shape
    re_im = np.dot(D, c).reshape(H, W) + X_mn
    plt.imshow(re_im, cmap=plt.gray())


if __name__ == '__main__':
    '''
    Read all images(grayscale) from jaffe folder and collapse each image
    to get an numpy array Ims with size (no_images, height*width).
    Make sure to sort the filenames before reading the images
    '''

    # Set correct CWD
    chdir(getcwd() + '/jaffe')

    # List to contain the names of the images
    I_Names = []

    # Add all names of images to the list
    for root, dirs, files in walk(getcwd()):
        for file in files:
            if '.tiff' in file:
                I_Names.append(file)

    I_Names.sort()
    I_Zero = Image.open(I_Names[0])
    (H, W) = I_Zero.size

    # Array to store the images
    Ims = np.empty(shape=(len(I_Names), H * W), dtype='float32')

    # Fill array with images
    for i in range(len(I_Names)):
        Im = np.array(Image.open(I_Names[i])).reshape(1, H * W)
        Ims[i] = np.float32(np.array(Im))

    # print Ims.shape

    # Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)
    # print X.shape

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''

    D = np.empty(shape=(H * W, 16), dtype='float32')
    Lambda = np.empty(shape=(16, ), dtype='float32')
    TS = 500
    Stop = 10 ** -8

    # Declare Theano symbolic variables
    X_Theano = T.matrix("X_Theano")
    d_i = T.vector("d_i")
    L = T.vector("L")
    d = T.matrix("d")
    acc = T.scalar("acc")

    for i in range(0, 16):
        # acc = np.empty(shape=(H * W, 1), dtype='float32')
        print 'i = ' + str(i)
        diff_c = 2
        diff_p = 0
        t = 0
        acc = 0  # Needs to be outside as A_i is before the while loop, Theano just changes d_i everytime

        y = np.float32(np.random.rand(H * W))
        D[:, i] = (y / np.linalg.norm(y))

        # Construct Theano function
        if i > 0:
            for j in range(0, i):
                print 'j = ' + str(j)
                acc += L[j] * T.pow(T.dot(d_i.T, d[:, j]), 2)
                # theano.printing.Print("Acc: ")(acc)
        # else:
        #     acc += L[0] * T.pow(T.dot(d_i.T, d_i), 2)
        Opt = - T.dot(T.dot(X_Theano, d_i).T, T.dot(X_Theano, d_i)) + acc
        gd = T.grad(Opt, d_i)
        f = theano.function([X_Theano, d_i, L, d], gd, on_unused_input='ignore')

        while (t == 0) or (t <= TS and abs(diff_c - diff_p) >= Stop):
            '''
            acc = np.empty(H * W)
            if i > 0:
                for j in range(0, i):
                    # print 'j = ' + str(j)
                    # acc += 2 * Lambda[j] * np.dot(np.dot(D[:, i].T, D[:, j]), D[:, j].T)
                    acc += 2 * Lambda[j] * np.dot(D[:, j], np.dot(D[:, j].T, D[:, i]))
                    # print np.linalg.norm(acc)
            else:
                acc = 0
            diff_p = diff_c
            # grad = - 2 * np.dot(np.dot(D[:, i].T, X.T), X) + acc
            grad = - 2 * np.dot(X.T, np.dot(X, D[:, i])) + acc
            print grad.shape, D[:, i].shape
            y = D[:, i] - 0.01 * grad
            '''

            # Theano
            diff_p = diff_c
            y = D[:, i] - 0.01 * f(X, D[:, i], Lambda, D)
            D[:, i] = y / np.linalg.norm(y)
            # print D[:, i].shape
            diff_c = np.dot(np.dot(X, D[:, i]).T, np.dot(X, D[:, i]))
            t += 1
            # if abs(diff_c - diff_p) < 500:
            #     print diff_c, diff_p, t
            Lambda[i] = diff_c

        print Lambda

    c = np.dot(D.T, X.T)

    for i in range(0, 200, 10):
        plot_mul(c, D, i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')

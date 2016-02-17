import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
# import matplotlib.cm as cm
from os import chdir, getcwd, walk
from PIL import Image
from theano.tensor.nnet.neighbours import images2neibs  # ,neibs2images

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''


def plot_mul(c, D, im_num, X_mn, num_coeffs, n_blocks):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Iterable
        an iterable with 9 elements representing the number_of coefficients
        to use for reconstruction for each of the 9 plots

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
            Dij = D[:, :nc]
            plt.subplot(3, 3, i * 3 + j + 1)
            plot(cij, Dij, n_blocks, X_mn, axarr[i, j])

    f.savefig(getcwd() + '/output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    # raise NotImplementedError
    # print '1'
    '''
    # Will have better resolution with the other method
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
    # plt.show() -> Doesn't work
    # Image does not work as well
    # Y = Image.fromarray(Stitch[0])
    # Y.show()

    # print '2'

    f, axarr = plt.subplots(4, 4)

    # print D.shape

    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i * 4 + j + 1)
            # print i * 4 + j + 1
            plt.imshow(D.T[i * 4 + j].reshape(sz, sz), cmap=plt.gray())

    # plt.show()
    plt.savefig(imname)

def plot(c, D, n_blocks, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        ax: the axis on which the image will be plotted
    '''
    # raise NotImplementedError
    # print (D.T).shape, c.shape
    H, W = X_mn.shape
    re_patch_T = np.dot(D, c) + X_mn.reshape(H * W, 1)
    # Without would result in many small faces
    # Had D_T dot X_T beforehand so have to retranpose
    re_patch = re_patch_T.T
    # print re_patch.shape
    # stream = theano.tensor.matrix('stream')
    # n2i = neibs2images(stream, (n_blocks, n_blocks), (256, 256))
    # inv_window_function = theano.function([stream], n2i)
    # plt.imshow(inv_window_function(re_patch), cmap=plt.gray())
    # re_im.show()

    Stitch = []
    for i in range(0, n_blocks):
        Stitch.append(re_patch[i*n_blocks].reshape(H, W))
        for j in range(1, n_blocks):
            Stitch[i] = np.hstack((Stitch[i], re_patch[i*n_blocks + j].reshape(H, W)))

    for i in range(1, len(Stitch)):
        Stitch[0] = np.vstack((Stitch[0], Stitch[i]))

    plt.imshow(Stitch[0], cmap=plt.gray())
    # plt.show()


def main():
    '''
    Read here all images(grayscale) from jaffe folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
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
    I_Array = np.empty(shape=(len(I_Names), H, W), dtype='float32')

    # Fill array with images
    for i in range(len(I_Names)):
        Im = Image.open(I_Names[i])
        I_Array[i] = np.float32(np.array(Im))

    szs = [16, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):

        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        '''

        # Defining variables
        images = T.tensor4('images')
        i2n = images2neibs(images, neib_shape=(sz, sz))

        # Constructing the Theano function
        window_function = theano.function([images], i2n)

        '''
        # Apply function to first image
        X = window_function(I_Array[0].reshape(1,1,H,W))

        # Apply function to remaining images and stack them to form X
        for i in range(1, len(I_Names)):
            Y = window_function(I_Array[i].reshape(1,1,H,W))
            X = np.vstack((X, Y))
        '''

        X = window_function(I_Array.reshape(len(I_Array), 1, H, W))

        X_mn = np.mean(X, 0)
        X = X - np.repeat(X_mn.reshape(1, -1), X.shape[0], 0)
        # print X.shape, X_mn.shape

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''

        X_T_X = np.dot(X.T, X)
        EigVal, EigVec = np.linalg.eigh(X_T_X)
        D = np.fliplr(EigVec)

        c = np.dot(D.T, X.T)  # As we know that D.T D = D D.T = I and (D.T D X.T).T = D D.T X

        # print D.shape, c.shape

        for i in range(0, 200, 10):
            plot_mul(c, D, i, X_mn.reshape((sz, sz)), num_coeffs=nc, n_blocks=int(256 / sz))

        plot_top_16(D, sz, imname=getcwd() + '/output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()

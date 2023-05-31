# pylint: skip-file
import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as tr

import numpy as np

from scipy.io import loadmat
import os

import pickle


def load_static_mnist(args):
    # set args
    print('loading static mnist')
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.is_binary = True
    args.dynamic_binarization = False

    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join(args.data_dir, 'MNIST_static', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(args.data_dir, 'MNIST_static', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(args.data_dir, 'MNIST_static', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')
    return x_train, x_val, x_test


def load_dynamic_mnist(args):
    # set args
    print('loading dynamic mnist')
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.is_binary = True
    args.dynamic_binarization = True

    # start processing
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader( datasets.MNIST(args.data_dir, train=True, download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                                batch_size=100, shuffle=True)

    test_loader = torch.utils.data.DataLoader( datasets.MNIST(args.data_dir, train=False,
                                                              transform=transforms.Compose([transforms.ToTensor()
                                                                        ])),
                                               batch_size=100, shuffle=True)

    # preparing data
    x_train = train_loader.dataset.data.float().numpy() / 255.
    x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )

    y_train = np.array( train_loader.dataset.targets.float().numpy(), dtype=int)

    x_test = test_loader.dataset.data.float().numpy() / 255.
    x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )

    y_test = np.array( test_loader.dataset.targets.float().numpy(), dtype=int)

    # validation set
    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    return x_train, x_val, x_test


def load_omniglot(args, n_validation=1345):
    print('loading omniglot')
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.is_binary = True
    args.dynamic_binarization = True

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')
    omni_raw = loadmat(os.path.join(args.data_dir, 'OMNIGLOT', 'chardata.mat'))

    # train and test data
    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # shuffle train data
    np.random.shuffle(train_data)

    # set train and validation data
    x_train = train_data[:-n_validation]
    x_val = train_data[-n_validation:]

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'
    return x_train, x_val, x_test


def load_caltech101silhouettes(args):
    print('loading caltech')
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.is_binary = True
    args.dynamic_binarization = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')
    caltech_raw = loadmat(os.path.join(args.data_dir, 'Caltech101Silhouettes', 'caltech101_silhouettes_28_split1.mat'))

    # train, validation and test data
    x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
    np.random.shuffle(x_train)
    x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
    np.random.shuffle(x_val)
    x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

    return x_train, x_val, x_test


def load_histopathologyGray(args):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'gray'
    args.is_binary = False
    args.vocab_size = 256
    args.dynamic_binarization = False

    # start processing
    with open(os.path.join(args.data_dir, 'HistopathologyGray/histopathology.pkl'), 'rb') as f:
        data = pickle.load(f, encoding="latin1")

    x_train = np.asarray(data['training']).reshape(-1, 28 * 28)
    x_val = np.asarray(data['validation']).reshape(-1, 28 * 28)
    x_test = np.asarray(data['test']).reshape(-1, 28 * 28)

    x_train = np.clip(x_train, 1./512., 1. - 1./512.)
    x_val = np.clip(x_val, 1./512., 1. - 1./512.)
    x_test = np.clip(x_test, 1./512., 1. - 1./512.)

    return x_train.astype(np.float32), x_val, x_test


def load_freyfaces(args, TRAIN = 1565, VAL = 200, TEST = 200):
    # set args
    args.input_size = [1, 28, 20]
    args.input_type = 'gray'
    args.is_binary = False
    args.vocab_size = 256
    args.dynamic_binarization = False

    # start processing
    # with open('datasets/Freyfaces/freyfaces.pkl', 'rb') as f:
    #     data = pickle.load(f, encoding="latin1")
    import scipy.io
    data = scipy.io.loadmat(os.path.join(args.data_dir, "Freyfaces/frey_rawface"))['ff'].T
    # data = (data + 0.5) / 256.
    data = data / 256.

    # shuffle data:
    np.random.shuffle(data)

    # train images
    x_train = data[0:TRAIN].reshape(-1, 28*20)
    # validation images
    x_val = data[TRAIN:(TRAIN + VAL)].reshape(-1, 28*20)
    # test images
    x_test = data[(TRAIN + VAL):(TRAIN + VAL + TEST)].reshape(-1, 28*20)

    return x_train.astype(np.float32), x_val, x_test


def load_raw_dataset(args):
    if args.dataset_name == 'static_mnist':
        x_train, x_val, x_test = load_static_mnist(args)
    elif args.dataset_name == 'dynamic_mnist':
        x_train, x_val, x_test = load_dynamic_mnist(args)
    elif args.dataset_name == 'omniglot':
        x_train, x_val, x_test = load_omniglot(args)
    elif args.dataset_name == 'caltech':
        x_train, x_val, x_test = load_caltech101silhouettes(args)
    elif args.dataset_name == 'histopathology':
        x_train, x_val, x_test = load_histopathologyGray(args)
    elif args.dataset_name == 'freyfaces':
        x_train, x_val, x_test = load_freyfaces(args)
    else:
        raise ValueError("unknown dataset %s" % args.dataset_name)
    # shuffle train data
    np.random.shuffle(x_train)
    print('# train', x_train.shape[0], '# valid', x_val.shape[0], '# test', x_test.shape[0])
    return x_train, x_val, x_test

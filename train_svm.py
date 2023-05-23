import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score

def main(args):
    logger = create_log()
    log_string(logger, args)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    train_data = train_set.data.reshape(-1, 28*28).numpy()[:6000]
    train_labels = train_set.targets.numpy()[:6000]

    test_data = test_set.data.reshape(-1, 28*28).numpy()[:6000]
    test_labels = test_set.targets.numpy()[:6000]

    if args.normalization == 'True':
        log_string(logger, 'Use normalization')
        train_data = normalization(train_data)
        test_data = normalization(test_data)

    if args.k != 0:
        log_string(logger, 'The value of k in PCA is %d' % (args.k))
        pca = PCA(n_components=args.k, svd_solver='randomized', random_state=42)
        train_data = pca.fit_transform(train_data)
        test_data = pca.transform(test_data)

    log_string(logger, 'The kernel is %s' % (args.kernel))
    log_string(logger, 'The classify strategy is %s' % (args.classify_strategy))
    log_string(logger, 'The value of C in SCM is %.2f' % (args.C))
    classifier = svm.SVC(C=args.C, kernel=args.kernel, decision_function_shape=args.classify_strategy)

    classifier.fit(train_data, train_labels)

    predict = classifier.predict(test_data)

    accuracy = accuracy_score(predict, test_labels)
    log_string(logger, 'The accuracy is %.3f' % (accuracy))

def visualize(imgs, reshape=True, imageSize=[28,28]):
    # Show sample faces using matplotlib
    imgs = imgs.T
    row = len(imgs) // 5
    fig, axes = plt.subplots(row, 5,sharex=True,sharey=True,figsize=(10, row*2))
    for i in range(len(imgs)):
        if reshape:
            img_ = imgs[i].reshape(imageSize)
        else:
            img_ = imgs[i]
        axes[i%row][i//row].imshow(img_, cmap="gray")
    plt.show()

def draw(x=None, y=None, tip=None, xlabel=None, ylabel=None, label_tip=None, fileName=None):
    plt.figure(facecolor='w',edgecolor='w')
    plt.rc('font',family='Times New Roman')
    if label_tip == None:
        plt.plot(x, y, linestyle = '-', linewidth = '1.5')
    else:
        for i in range(len(tip)):
            plt.plot(x[i,:], linestyle = '-', linewidth = '1.5', label=label_tip+str(tip[i]))
        plt.legend()
    plt.xlabel(xlabel, fontsize='x-large')
    plt.ylabel(ylabel, fontsize='x-large')
    plt.grid()
    # plt.show()
    if fileName != None:
        plt.savefig('./hidden_test/'+fileName+'.png', dpi=600, format='png')

def normalization(images):
    mean = np.mean(images, axis=0, keepdims=True)
    images = (images - mean)/np.var(images)
    return images

def parse_args():
    parser = argparse.ArgumentParser(description='Numply MLP example')
    parser.add_argument('--k', type=int, default=470, metavar='N', help='The value of K in PCA (default: 0)')
    parser.add_argument('--normalization', type=str, default='False', metavar='STR', help='Use the normalization (default: True)')
    parser.add_argument('--classify-strategy', type=str, default='ovr', metavar='STR', help='Use the normalization (default: True)')
    parser.add_argument('--kernel', type=str, default='rbf', metavar='STR', help='The SVM kernel (default: rbf)')
    parser.add_argument('--C', type=float, default=2.0, metavar='N', help='The value of C in SVM (default: 1.0)')
    return parser.parse_args()

def create_log():
    log_dir = Path('./svm/')
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("SVM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s_log.txt' % (log_dir, "svm")) 
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def log_string(logger, str):
    logger.info(str)
    print(str)

if __name__ == '__main__':
    args = parse_args()
    main(args)
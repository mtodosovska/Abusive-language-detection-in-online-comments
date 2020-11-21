import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(matrices, titles, classes):

    font = {'fontname': 'Ariel', 'size': 24}
    font_i = {'fontname': 'Ariel', 'size': 30}
    for matrix, title in zip(matrices, titles):
        fig = plt.figure()
        fig.set_size_inches(8, 6)
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title, weight='bold', **font)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, **font)
        plt.yticks(tick_marks, classes, **font)

        fmt = 'd'
        thresh = matrix.max() / 1.1
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black", **font_i)

        plt.tight_layout()
        plt.savefig('../results/confusion matrices/final.jpg')
        plt.show()


if __name__ == '__main__':

    classes = ['Class 0', 'Class 1']
    confusion_matrix = np.array([[1815,  375], [ 468, 1673]])

    plot_confusion_matrix([confusion_matrix], ['Confusion Matrix'], classes)


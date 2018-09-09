import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D


def knn_classifier(k, predict, data_set, label):
    if predict.ndim is 1:
        predict = predict.reshape(1, -1)
    assert len(predict[0]) is len(data_set[0]), 'Match dimension between dataset and item to predict'
    assert len(data_set) is len(label), 'Size of data_set and size of label must be same'
    for pos in predict:
        distance = np.array([np.linalg.norm(pos - data) for data in data_set])
        result, counter = collections.Counter(label[np.argsort(distance)[:k]]).most_common()[0]
        yield result, counter / k


def test_knn_classifier(k):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    x, y = load_iris(True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    result_l = np.array([i[0] for i in knn_classifier(k, x_test, x_train, y_train)])
    print('{:.2f} % correct'.format(np.sum(y_test == result_l) * 100 / len(result_l)))


def visualize_data(predict, data_set, label):
    if predict.ndim is 1:
        predict = predict.reshape(1, -1)
    if data_set.ndim is 1:
        data_set = data_set.reshape(1, -1)
    dim_of_data = len(data_set[0])
    assert 1 < dim_of_data < 4, 'This dimension cannot be visualized'
    print(predict[0], dim_of_data)
    assert len(predict[0]) is dim_of_data, 'Dimension of prediction and  data set must be same'

    unique_label = np.unique(label)
    color_dict = dict(zip(unique_label, cm.rainbow(np.linspace(0, 1, len(unique_label)))))
    fig = plt.figure()
    fig.add_subplot(111, projection=None if dim_of_data is 2 else '3d')
    data = map(np.ndarray.flatten, np.hsplit(data_set, dim_of_data))
    predict_ = map(np.ndarray.flatten, np.hsplit(predict, dim_of_data))
    color = list(map(lambda x: color_dict[x], np.ndarray.flatten(label)))
    plt.scatter(*data, c=color)
    plt.scatter(*predict_, c='k')
    plt.show()

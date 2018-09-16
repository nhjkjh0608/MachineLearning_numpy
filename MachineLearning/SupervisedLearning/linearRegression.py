import numpy as np
import matplotlib.pyplot as plt
import sys


def liner_regression_gradient_descent(x, y, alpha=0.0005, initial_theta=None, iter_num=1000, stream=sys.stdout):
    if x.ndim is 1:
        x = x.reshape(1, -1).transpose()
    assert len(x) is len(y)
    data_num = len(x)
    new_x = np.hstack((np.ones(data_num).reshape(-1, 1), x))
    trans_new_x = new_x.transpose()
    initial_theta = np.ones(len(x[0]) + 1) if initial_theta is None else initial_theta
    d_alpha = alpha / iter_num
    cost_saver = []
    for _ in range(iter_num):
        loss = np.dot(new_x, initial_theta) - y
        cost = np.sum(loss ** 2) / (2 * data_num)
        gradient = np.dot(trans_new_x, loss)
        initial_theta = initial_theta - alpha * gradient
        alpha -= d_alpha
        cost_saver.append(str(cost)+'\n')
    if stream is not None:
        for i in cost_saver:
            stream.write(i)
    return initial_theta, float(cost_saver[-1])


def linear_regression_ordinary_least_squares(x, y):
    if x.ndim is 1:
         x = x.reshape(-1, 1)
    assert len(x) is len(y)
    x = np.hstack((np.ones(len(x)).reshape(-1, 1), x))
    trans_x = np.transpose(x)

    y = y.reshape(-1, 1)
    trans_y = np.transpose(y)
    m, n = x.shape
    middle_ = np.dot(np.linalg.pinv(np.dot(trans_x,x)), trans_x)
    thetha = np.dot(middle_, y).flatten()

    p = np.dot(x, middle_)
    trans_p = np.transpose(p)
    l = np.eye(m) - np.ones((m, m)) / m

    upper = np.linalg.multi_dot((trans_y, trans_p, l, p, y))
    lower = np.linalg.multi_dot((trans_y, l, y))
    r_2 = (upper/lower)[0][0]
    return thetha, r_2


def linear_regression_generalized_least_squares(x, y, cov_matrix=None):
    if x.ndim is 1:
        x = x.reshape(-1, 1)
    assert len(x) is len(y)
    y = y.reshape(-1, 1)
    x = np.hstack((np.ones(len(x)).reshape(-1, 1), x))
    trans_x = np.transpose(x)
    if cov_matrix is None:
        cov_matrix = np.eye(len(x))
    v = cov_matrix
    inv_v = np.linalg.pinv(v)
    middle_ = np.dot(trans_x, v)
    theta = np.linalg.multi_dot((np.linalg.pinv(np.dot(middle_, x)), trans_x, inv_v, y))
    return theta.flatten()


def test_linear_regression():
    pass


def visualize_data():
    pass


def make_data():
    pass



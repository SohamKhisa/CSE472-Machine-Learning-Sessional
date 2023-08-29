"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
import numpy as np

def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    # y_true == y_pred => TP + TN
    # len(y_true) = TP + TN + FP + FN
    accuracy = np.sum(np.equal(y_true, y_pred)) / len(y_true)
    return accuracy
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    precision = tp / (tp + fp)
    return precision


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    recall = tp / (tp + fn)
    return recall


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    precission = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * precission * recall / (precission + recall)
    # todo: implement
    return f1

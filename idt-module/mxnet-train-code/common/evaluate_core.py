import io
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


def to_onehot(y_true, num_classes=None):
    y_true = np.array(y_true, dtype='int')
    input_shape = y_true.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y_true = y_true.ravel()
    if not num_classes:
        num_classes = np.max(y_true) + 1
    n = y_true.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y_true] = 1
    output_shape = input_shape + (num_classes, )
    categorical = np.reshape(categorical, output_shape)
    return categorical


class ClassificationEvaluate(object):

    def __init__(self, y_onehot: list, y_score: list, average='micro'):
        self.y_onehot = y_onehot
        self.y_score = np.array(y_score)
        self.average = average

        self.y_pred = np.array(
            [np.argwhere(item == max(item)) for item in y_score]).reshape(-1)
        self.y_true = np.array(
            [np.argwhere(item == max(item)) for item in y_onehot]).reshape(-1)

        assert len(self.y_pred) == len(self.y_onehot), '模型预测和gt图片数量不一致'

    def get_precision(self, class_labels=None) -> float:
        return metrics.precision_score(
            self.y_true,
            self.y_pred,
            labels=class_labels,
            average=self.average)

    def get_recall(self, class_labels=None) -> float:
        return metrics.recall_score(
            self.y_true,
            self.y_pred,
            labels=class_labels,
            average=self.average)

    def get_f1(self, class_labels=None) -> float:
        return metrics.f1_score(
            self.y_true,
            self.y_pred,
            labels=class_labels,
            average=self.average)

    def get_mAP(self) -> float:
        AP_list = []
        for i in range(len(self.y_score)):
            AP_list.append(
                metrics.average_precision_score(self.y_onehot[i],
                                                self.y_score[i]))
        return np.mean(AP_list)

    def get_multi_classes_roc_line(self, class_labels=None):
        if not class_labels:
            class_labels = list(set(self.y_true))

        fig = plt.figure()
        for i in class_labels:
            fpr, tpr, thresholds = metrics.roc_curve(self.y_onehot[:, i],
                                                     self.y_score[:, i])
            auc_value = metrics.auc(fpr, tpr)
            axs = plt.plot(
                fpr,
                tpr,
                label='class: {}: auc (area = {:.3f})'.format(i, auc_value))
            plt.plot(
                fpr, thresholds, label='class: {}: threshold line'.format(i))

        axs[0].axes.set_xscale('log')
        axs[0].axes.grid(True)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Classification ROC curve')
        plt.legend(loc='best')

        buffer = io.BytesIO()
        fig.canvas.print_png(buffer)
        data = buffer.getvalue()
        # plt.savefig(save_path)
        plt.close()
        buffer.close()
        return data

    def get_roc_line(self, class_labels=None):
        y_score_list, y_valid_list = [], []
        if not class_labels:
            class_labels = list(set(self.y_true))

        for index, gt in enumerate(self.y_true):
            if gt in class_labels:
                y_valid_list.append(1)
            else:
                y_valid_list.append(0)

            y_score_list.append(max(self.y_score[index][class_labels]))

        fig = plt.figure()
        fpr, tpr, thresholds = metrics.roc_curve(y_valid_list, y_score_list)
        auc_value = metrics.auc(fpr, tpr)
        axs = plt.plot(fpr, tpr, label='auc (area = {:.3f})'.format(auc_value))
        plt.plot(fpr, thresholds, label='threshold line')

        axs[0].axes.set_xscale('log')
        axs[0].axes.grid(True)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Classification ROC curve')
        plt.legend(loc='best')

        buffer = io.BytesIO()
        fig.canvas.print_png(buffer)
        data = buffer.getvalue()
        # plt.savefig(save_path)
        plt.close()
        buffer.close()
        return data

    def get_multi_classes_pr_line(self, class_labels=None):
        if not class_labels:
            class_labels = list(set(self.y_true))

        fig = plt.figure()
        for i in class_labels:
            precision, recall, _ = metrics.precision_recall_curve(
                self.y_onehot[:, i], self.y_score[:, i])
            plt.plot(recall, precision, label='class: {}: PR line'.format(i))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')

        buffer = io.BytesIO()
        fig.canvas.print_png(buffer)
        data = buffer.getvalue()
        # plt.savefig(save_path)
        plt.close()
        buffer.close()
        return data

    def get_pr_line(self, class_labels=None):
        y_score_list, y_valid_list = [], []
        if not class_labels:
            class_labels = list(set(self.y_true))

        for index, gt in enumerate(self.y_true):
            if gt in class_labels:
                y_valid_list.append(1)
            else:
                y_valid_list.append(0)

            y_score_list.append(max(self.y_score[index][class_labels]))

        precision, recall, _ = metrics.precision_recall_curve(
            y_valid_list, y_score_list)

        fig = plt.figure()
        plt.plot(recall, precision, label='PR line')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')

        buffer = io.BytesIO()
        fig.canvas.print_png(buffer)
        data = buffer.getvalue()
        # plt.savefig(save_path)
        plt.close()
        buffer.close()
        return data

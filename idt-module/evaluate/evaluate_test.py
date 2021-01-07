import cv2
import io
import numpy as np
from evaluate_core import ClassificationEvaluate
from PIL import Image
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# 加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
# 将标签二值化
y = label_binarize(y, classes=[0, 1, 2])
# 设置种类
n_classes = y.shape[1]

# 训练模型并预测
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.5, random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(
    svm.SVC(kernel='linear', probability=True, random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_train)


def softmax(x):
    """ softmax function """
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x


y_score = softmax(y_score)
y_true = y_train

c_evaluate = ClassificationEvaluate(y_true, y_score)

# 预测所有类别precision
print('precision: ', c_evaluate.get_precision())

# 指定0,1,2,3四个类别的precision
print('precision: ', c_evaluate.get_precision(class_labels=[0, 1]))

# 预测所有类别recall
print('recall: ', c_evaluate.get_recall())

# 指定0,1,2,3四个类别的recall
print('recall: ', c_evaluate.get_recall(class_labels=[0, 1]))

# 预测所有类别f1
print('f1: ', c_evaluate.get_f1())

# 指定0,1,2,3四个类别的f1
print('f1: ', c_evaluate.get_f1(class_labels=[0, 1]))

# 预测mAP
print('mAP: ', c_evaluate.get_mAP())

buffer = io.BytesIO()
data = c_evaluate.get_multi_classes_roc_line(class_labels=[0, 1, 2])
print(data)
buffer.write(data)
img = Image.open(buffer)
img = np.asarray(img)
cv2.imwrite('./01.jpg', img)
buffer.close()

buffer = io.BytesIO()
data = c_evaluate.get_roc_line(class_labels=[1])
print(data)
buffer.write(data)
img = Image.open(buffer)
img = np.asarray(img)
cv2.imwrite('./02.jpg', img)
buffer.close()

buffer = io.BytesIO()
data = c_evaluate.get_multi_classes_pr_line(class_labels=[0, 1, 2])
print(data)
buffer.write(data)
img = Image.open(buffer)
img = np.asarray(img)
cv2.imwrite('./03.jpg', img)
buffer.close()

buffer = io.BytesIO()
data = c_evaluate.get_pr_line(class_labels=[1])
print(data)
buffer.write(data)
img = Image.open(buffer)
img = np.asarray(img)
cv2.imwrite('./04.jpg', img)
buffer.close()

import numpy as np
from sklearn.metrics import confusion_matrix

rest_train_file_path = 'rest_train.txt'
rest_train = np.loadtxt(rest_train_file_path)
ytrain_pred = rest_train[:,0:12]
ytrain_true = rest_train[:,12:]

ytrain_pred = ytrain_pred.argmax(axis=1)
ytrain_true = ytrain_true.argmax(axis=1)

print(confusion_matrix(ytrain_true, ytrain_pred))


rest_test_file_path = 'rest_test.txt'
rest_test = np.loadtxt(rest_test_file_path)
ytest_pred = rest_test[:, 0:12]
ytest_true = rest_test[:, 12:]
ytest_pred = ytest_pred.argmax(axis=1)
ytest_true = ytest_true.argmax(axis=1)

print(confusion_matrix(ytest_true, ytest_pred))



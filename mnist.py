'''
Just reviewing classification with the classic dataset
'''

from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler


def fetch_data():
    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]
    return X, y

# show any digit in the dataset index 1-70000
def show_digit(digit):
    some_digit = X[digit]
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
                    interpolation = "nearest")
    plt.axis("off")
    plt.show()
'''
ex. input:
plt.figure(figsize=(7,7))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
'''
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")
    plt.show()

def train_test_split():
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    return X_train, X_test, y_train, y_test

def number_train_sgd(number):
    y_train_num = (y_train == number)
    y_test_num = (y_test == number)
    sgd_clf = SGDClassifier(random_state = 34)
    # model = sgd_clf.fit(X_train, y_train_num)
    '''
    # implementing cross validation
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train_num):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train_num[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train_num[test_index])

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))
    '''
    #or just using the built in
    scores = cross_val_score(sgd_clf, X_train, y_train_num, cv=3, scoring="f1")
    print(scores)
    y_train_predict = cross_val_predict(sgd_clf, X_train, y_train_num, cv=3)
    confusion = confusion_matrix(y_train_num, y_train_predict)
    print("[[ TN  FP]]")
    print("[ FN  TP]]")
    print(confusion)
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_num, cv=3, method="decision_function")
    #precision/recall plot
    precisions, recalls, thresholds = precision_recall_curve(y_train_num, y_scores[:,1])
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    plt.savefig("plots/precision_recall.png")
    #roc curve
    fpr, tpr, thresholds = roc_curve(y_train_num, y_scores[:,1])
    auc_score = roc_auc_score(y_train_num, y_scores[:,1])
    print(auc_score)
    return fpr, tpr

def number_train_rf(number):
    y_train_num = (y_train == number)
    y_test_num = (y_test == number)
    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_num, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:,1]
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_num, y_scores_forest)
    return fpr_forest, tpr_forest

def plot_roc():
    plt.figure(figsize=(8, 6))
    plt.plot(sgd_fpr, sgd_tpr, linewidth=2, label='ROC Curve')
    plt.plot(fpr_forest, tpr_forest, linewidth=2, label='Random Forest')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="bottom right")
    plt.savefig("plots/roc_curve_plot")

def sgd_ova(digit):
    some_digit = X[digit]
    sgd_clf = SGDClassifier(random_state = 34)
    sgd_clf.fit(X_train, y_train)
    prediction = sgd_clf.predict([some_digit])
    some_digit_scores = sgd_clf.decision_function([some_digit])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    cvs = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
    print(cvs)
    print(some_digit_scores)
    print(prediction)

def sgd_ovo(digit):
    some_digit = X[digit]
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state = 34))
    ovo_clf.fit(X_train, y_train)
    prediction = ovo_clf.predict([some_digit])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    cvs = cross_val_score(ovo_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
    print(cvs)
    print(prediction)

def rf(digit):
    some_digit = X[digit]
    forest_clf = RandomForestClassifier(random_state=42)
    forest_clf.fit(X_train, y_train)
    prediction = forest_clf.predict([some_digit])
    confidences = forest_clf.predict_proba([some_digit])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    # cvs = cross_val_score(forest_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
    # print(cvs)
    print(prediction)
    y_train_pred = cross_val_predict(forest_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)
    # plt.matshow(conf_mx, cmap=plt.cm.gray)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    plt.close("all")
    X, y = fetch_data()
    # show_digit(32000)
    # plot_digits(example_images, images_per_row=10)
    X_train, X_test, y_train, y_test = train_test_split()
    # sgd_fpr, sgd_tpr = number_train_sgd(5)
    # fpr_forest, tpr_forest = number_train_rf(5)
    # plot_roc()
    sgd_ova(3600)
    sgd_ovo(3600)
    rf(32000)

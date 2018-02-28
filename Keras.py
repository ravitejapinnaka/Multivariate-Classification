import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras import optimizers
from itertools import cycle
import itertools


def plot_confusion_matrix(test_y, pred_y, class_names, filename):
    """
    This function prints and plots the confusion matrix.
    """
    cmap = plt.cm.Blues
    # Compute confusion matrix
    cm = confusion_matrix(
        np.argmax(test_y, axis=1), np.argmax(pred_y, axis=1))
    np.set_printoptions(precision=2)
    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("LSTM Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename + ".png")
    # Plot normalized confusion matrix


def plotroc(test_y, pred_y, n_classes, filename):
    """
    Compute ROC curve and ROC area for each class
    """
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(test_y[:, i], pred_y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot all ROC curves
    plt.figure()
    lw = 1
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'''.
                 format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LSTM Performance')
    plt.legend(loc="lower right")
    plt.savefig(filename + '.png')
    return


def read_data(file_path):
    data = pd.read_excel(file_path, header=0)
    return data


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size - 1
        start += (window_size)


def extract_segments(data, window_size):
    segments = None
    labels = np.empty((0))
    for (start, end) in windows(data, window_size):
        if (len(data.ix[start:end]) == (window_size)):
            signal = np.asarray(data.ix[start:end])[:,2:16]
            if segments is None:
                segments = signal
            else:
                segments = np.vstack([segments, signal])
            labels = np.append(labels, data.ix[start:end]["goal"][start])
    return segments, labels


if __name__ == '__main__':

    """Hyperparameters"""
    win_size = 89
    num_var = 14
    split_ratio = 0.8

    """Load data:
    segment: Each time series of 89 samples is named as segment.
    label: Each segment is associated with a label
    """
    data = read_data("Multi-variate-Time-series-Data.xlsx")
    segments, labels = extract_segments(data, win_size)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
    reshaped_segments = segments.reshape(
        [int(len(segments) / (win_size)), (win_size), num_var])

    """Create Train and Test Split based on split ratio"""

    train_test_split = np.random.rand(len(reshaped_segments)) < split_ratio
    train_x = reshaped_segments[train_test_split]
    train_y = labels[train_test_split]
    test_x = reshaped_segments[~train_test_split]
    test_y = labels[~train_test_split]

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(128, input_dim=num_var, input_length=win_size, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                          epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Fit the network
    model.fit(train_x, train_y, nb_epoch=48, batch_size=149,
              verbose=2, validation_split=0.1)

    # Predict Test Data and Plot ROC
    pred_y = model.predict(test_x, batch_size=64, verbose=2)
    plotroc(test_y, pred_y, 3, 'test_roc')
    class_names = ['Class 0', 'Class 1', 'Class 3']
    plot_confusion_matrix(test_y, pred_y, class_names, 'test_conf')

    pred_y = model.predict(train_x, batch_size=64, verbose=2)
    plotroc(train_y, pred_y, 3, 'train_roc')
    plot_confusion_matrix(train_y, pred_y, class_names, 'train_conf')

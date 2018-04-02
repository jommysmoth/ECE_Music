"""
First Attempt at Classification.

Using basic image classification to check accuracy on labels
"""
import create_spec_train
import create_spec_test
import numpy as np
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    label_dict, labels = create_spec_train.main()
    clf = MLPClassifier(solver='sgd', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    # Label Dictionary:Set Amount Per Label-Data_X-Data_Y-Channels
    x = []
    y = []
    for ind, label in enumerate(labels):
        train_set = label_dict[label]
        for samp in range(train_set.shape[0]):
            for chn in range(train_set.shape[3]):
                examp = train_set[samp, :, :, chn]
                examp = np.ravel(examp)
                x.append(examp)
                y.append(ind)
    clf.fit(x, y)
    print(clf.predict(x))
    print(y)


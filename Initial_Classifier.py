"""
First Attempt at Classification.

Using basic image classification to check accuracy on labels
"""
import create_spec_train as cst
import numpy as np

if __name__ == '__main__':
    labels = ['Experimental Rock', 'Grindcore', 'Hardcore', 'Indie Rock', 'Post Rock']
    procd = cst.ProcessingData(labels, train_amount=0.8)
    train, test = procd.rand_train_test_main()
    # Label Dictionary:Set Amount Per Label-Data_X-Data_Y-Channels

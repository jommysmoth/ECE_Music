"""Munching."""
import pickle
import numpy as np
if __name__ == '__main__':
    labels = ['Jazz', 'Rock', 'Rap']  # Have program output this soon
    cut_amount = 10
    external_file_area = '/media/jommysmoth/Storage/ECE_DATA/data'
    for lab in labels:
        with open(external_file_area + '_dict/' + lab + '.pickle', 'rb') as handle:
            full_train = pickle.load(handle)
        len_full = len(full_train[lab])
        for cut in range(cut_amount):
            start = int(cut * (len_full / cut_amount))
            end = start + int(len_full / cut_amount)
            if end > len_full:
                end = len_full
            new_cut = full_train[lab][start:end]
            fin_cut = []
            for examp in new_cut:
                low_val = np.amin(examp)
                translated = examp + np.absolute(low_val)
                translated /= np.mean(translated)
                fin_cut.append(translated)
            with open(external_file_area + '_dict/' + lab + str(cut) + '.pickle', 'wb') as handle:
                pickle.dump(fin_cut, handle, protocol=pickle.HIGHEST_PROTOCOL)
        full_train = None

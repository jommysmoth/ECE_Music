"""Munching."""
import pickle
if __name__ == '__main__':
    labels = ['Jazz', 'Rock', 'Rap', 'Electronic', 'Classical', 'Folk']  # Have program output this soon
    train_test = ['_test', '_train']
    cut_amount = 5
    external_file_area = '/media/jommysmoth/Storage/ECE_DATA/data'
    for tt in train_test:
        for lab in labels:
            with open(external_file_area + '_dict/' + lab + tt + '.pickle', 'rb') as handle:
                full = pickle.load(handle)
            len_full = len(full)
            for cut in range(cut_amount):
                start = int(cut * (len_full / cut_amount))
                end = start + int(len_full / cut_amount)
                if end > len_full:
                    end = len_full
                new_cut = full[start:end]
                with open(external_file_area + '_dict/' + lab + tt + str(cut) + '.pickle', 'wb') as handle:
                    pickle.dump(new_cut, handle, protocol=pickle.HIGHEST_PROTOCOL)
            full_train = None

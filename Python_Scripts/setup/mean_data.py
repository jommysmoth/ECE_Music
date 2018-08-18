"""
Creating Mean Value.

Goes Through Training Set For Now
"""
import numpy as np
import create_spec as cst
import pickle
import os


def delete_load_folder(path):
    """Used for saving space on songs for loading."""
    song_list = os.listdir(path)
    for name in song_list:
        os.remove(path + '/' + name)
    print('Old Data Erased')
    return


if __name__ == '__main__':
    labels = ['Jazz', 'Rock', 'Rap']
    override_convert = False
    update_songs = 1000
    external_file_area = '/media/jommysmoth/Storage/ECE_DATA/data'
    procd = cst.ProcessingData(labels, train_amount=1,
                               seconds_total=30,
                               data_folder='/media/jommysmoth/Storage/ECE_DATA/data',
                               override_convert=override_convert,
                               conversions=update_songs,
                               ext_storage=external_file_area)
    if False:
        train, test = procd.main_train_test()
        with open('mean_data/train.pickle', 'wb') as handle:
            pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Mean Set Saved Externally')
        exit()
        delete_load_folder('/media/jommysmoth/Storage/ECE_DATA/data_wav')

    else:
        with open('mean_data/train.pickle', 'rb') as handle:
            dset = pickle.load(handle)
    # plt.imshow(dset['Jazz'][10], cmap=cm.inferno)
    major_list = []
    for lab in labels:
        major_num = np.sum(dset[lab], axis=0) / len(dset[lab])
        major_list.append(np.sum(major_num) / (major_num.shape[0] * major_num.shape[1]))
    mean_all_labels = np.array(major_list)

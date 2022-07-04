import numpy as np
from scipy.fftpack import fft
from scipy.signal import filtfilt
from sklearn.decomposition import PCA, FastICA
from scipy.signal.filter_design import butter
from mne.decoding import UnsupervisedSpatialFilter
import mne


def get_pca_data(data, random_seed=0):
    components = data.shape[1]

    pca = UnsupervisedSpatialFilter(PCA(components, random_state=random_seed), average=False)
    pca_data = pca.fit_transform(data)

    return pca_data


def get_ica_data(data, random_seed=0):
    components = data.shape[1]

    ica = UnsupervisedSpatialFilter(FastICA(components, random_state=random_seed), average=False)
    ica_data = ica.fit_transform(data)

    return ica_data

def process_BCICIV_data_set_1(data, low_frequency_limit, high_frequency_limit, random_seed=0):
    sample_rate = data['nfo'][0][0]['fs'][0][0]
    nfo_channel_list = data['nfo'][0][0]['clab'][0].tolist()

    channels = ['C3', 'Cz', 'C4']
    select_channel = []
    for channel in channels:
        select_channel.append(nfo_channel_list.index([channel]))
		
    pos_list = data['mrk'][0][0]['pos'][0]
    y = data['mrk'][0][0]['y'][0]
    motor_imagery_data = []
    motor_imagery_data_labels = []
    for index, pos in enumerate(pos_list):
        mi_data = data['cnt'][pos + 2 * sample_rate:pos + 6 * sample_rate]
        if mi_data.shape[0] != 400:
            continue
        motor_imagery_data.append(mi_data)
        motor_imagery_data_labels.append(y[index])
    motor_imagery_data = np.array(motor_imagery_data)
    motor_imagery_data_labels = np.array(motor_imagery_data_labels)

    select_channel_data = []
    for d in motor_imagery_data:
        test = d.T
        temp = []
        for i in select_channel:
            temp.append(d.T[i])
        select_channel_data.append(np.array(temp))

    select_channel_data = np.array(select_channel_data, dtype="float32")
    select_channel_data = select_channel_data * 0.1

    if low_frequency_limit <= 0:
        low_frequency_limit = 1
    if high_frequency_limit >= sample_rate / 2:
        high_frequency_limit = sample_rate / 2 - 1

    butter_order = 3
    Wn = 2 * np.array([low_frequency_limit, high_frequency_limit]) / sample_rate

    b, a = butter(N=butter_order, Wn=Wn, btype='bandpass', output='ba')

    filter_data = []
    for d in select_channel_data:
        temp = []
        for i in d:
            temp.append(np.array(filtfilt(b, a, i)))
        filter_data.append(np.array(temp))
    filter_data = np.array(filter_data)

    ica_filter_data = get_ica_data(filter_data)
    pca_filter_data = get_pca_data(filter_data)
    labels = np.array([label if label != -1 else 0 for label in motor_imagery_data_labels])

    return ica_filter_data, np.array([x if x != -1 else 0 for x in y])


def process_BCICIII_data_set_IVa(data, low_frequency_limit, high_frequency_limit, tra_nums, random_seed=0):
    sample_rate = data['nfo'][0][0]['fs'][0][0]
    nfo_channel_list = data['nfo'][0][0]['clab'][0].tolist()

    channels = ['C3', 'Cz', 'C4']
    select_channel = []
    for channel in channels:
        select_channel.append(nfo_channel_list.index([channel]))

    pos_list = data['mrk'][0][0]['pos'][0]
    y = data['mrk'][0][0]['y'][0]
    motor_imagery_data = []
    motor_imagery_data_labels = []
    for index, pos in enumerate(pos_list):
        mi_data = data['cnt'][pos:pos + 3 * sample_rate]
        # if mi_data.shape[0] != 400:
        #     continue
        motor_imagery_data.append(mi_data)
        motor_imagery_data_labels.append(y[index])
    motor_imagery_data = np.array(motor_imagery_data)
    motor_imagery_data_labels = np.array(motor_imagery_data_labels)

    motor_imagery_data = motor_imagery_data[0:tra_nums]
    motor_imagery_data_labels = motor_imagery_data_labels[0:tra_nums]

    select_channel_data = []
    for d in motor_imagery_data:
        test = d.T
        temp = []
        for i in select_channel:
            temp.append(d.T[i])
        select_channel_data.append(np.array(temp))

    select_channel_data = np.array(select_channel_data, dtype="float32")
    select_channel_data = select_channel_data * 0.1

    if low_frequency_limit <= 0:
        low_frequency_limit = 1
    if high_frequency_limit >= sample_rate / 2:
        high_frequency_limit = sample_rate / 2 - 1

    butter_order = 3
    Wn = 2 * np.array([low_frequency_limit, high_frequency_limit]) / sample_rate

    b, a = butter(N=butter_order, Wn=Wn, btype='bandpass', output='ba')

    filter_data = []
    for d in select_channel_data:
        temp = []
        for i in d:
            temp.append(np.array(filtfilt(b, a, i)))
        filter_data.append(np.array(temp))
    filter_data = np.array(filter_data)

    ica_filter_data = get_ica_data(filter_data)
    labels = np.array([label if label != 2 else 0 for label in motor_imagery_data_labels])

    return ica_filter_data, labels
    pass


def filter_data(data):
    butter_order = 3
    Wn = 2 * np.array([8, 30]) / 250

    b, a = butter(N=butter_order, Wn=Wn, btype='bandpass', output='ba')

    filter_data = []
    for d in data:
        temp = []
        for i in d:
            temp.append(np.array(filtfilt(b, a, i)))
        filter_data.append(np.array(temp))
    filter_data = np.array(filter_data)
    return filter_data


def process_BCICIV_data_set_2b(rawDataGDF, random_seed=0):
    eventDescription = {'276': "eyesOpen", '277': "eyesClosed", '768': "startTrail", '769': "cueLeft",
                        '770': "cueRight", '781': "feedback", '783': "cueUnknown",
                        '1023': "rejected", '1077': 'horizonEyeMove', '1078': "verticalEyeMove",
                        '1079': "eyeRotation", '1081': "eyeBlinks", '32766': "startRun"}

    ch_types = ['eeg', 'eeg', 'eeg']
    ch_names = ['EEG_Cz', 'EEG_C3', 'EEG_C4']
    label_name = ["left", "right"]
    time_label = ["4_55", "55_7"]

    info = mne.create_info(ch_names=ch_names, sfreq=rawDataGDF.info['sfreq'], ch_types=ch_types)

    data = np.squeeze(np.array(
        [rawDataGDF['EEG:Cz'][0], rawDataGDF['EEG:C3'][0], rawDataGDF['EEG:C4'][0]]))

    rawData = mne.io.RawArray(data, info)

    event, _ = mne.events_from_annotations(rawDataGDF)
    event_id = {}
    for i in _:  
        event_id[eventDescription[i]] = _[i]

    epochs4_55 = mne.Epochs(rawData, event, event_id, tmin=4, tmax=5.5, baseline=None,
                            event_repeated='merge')
    epochs55_7 = mne.Epochs(rawData, event, event_id, tmin=5.5, tmax=7, baseline=None,
                            event_repeated='merge')
    left_data, right_data = None, None

    for time in time_label:
        if time == "4_55":
            epochs = epochs4_55
        if time == "55_7":
            epochs = epochs55_7


        ev_left = epochs['cueLeft']
        ev_right = epochs['cueRight']
        left_data = ev_right.get_data()
        right_data = ev_left.get_data()

        left_data = np.array(left_data)
        right_data = np.array(right_data)

    filter_left_data = filter_data(left_data)
    filter_right_data = filter_data(right_data)

    ica_filter_left_data = get_ica_data(filter_left_data)
    ica_filter_right_data = get_ica_data(filter_right_data)

    label_left = np.ones(len(ica_filter_left_data))
    label_right = np.zeros(len(ica_filter_right_data))

    ica_filter_data = np.append(ica_filter_left_data, ica_filter_right_data, axis=0)
    labels = np.append(label_left, label_right, axis=0)

    return ica_filter_data, labels


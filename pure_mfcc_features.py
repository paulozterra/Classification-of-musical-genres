import numpy as np
import pandas
import os
import sklearn
import librosa
from sklearn.preprocessing import MinMaxScaler



def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_sample_arrays(dataset_dir, folder_name, samp_rate):
    path_of_audios = librosa.util.find_files(dataset_dir + "/" + folder_name)
    audios = []
    for audio in path_of_audios:
        x, sr = librosa.load(audio, sr=samp_rate, duration=3.0)
        audios.append(x)
    audios_numpy = np.array(audios)
    return audios_numpy

def extract_features(signal, sample_rate, frame_size, hop_size):
    mel = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    mel=np.array(mel)
    return(mel)

samp_rate = 22050
frame_size = 2048
hop_size = 512
window=23

dataset_dir = "C:/Users/PAULO/Documents/GitHub/Music-Classifier/FMA_WAVs"

sub_folders = get_subdirectories(dataset_dir)

labels = []

print("Extracting features from audios...")
i=0
fulldata=[]
for sub_folder in sub_folders:
    print(".....Working in folder:", sub_folder)
    sample_arrays = get_sample_arrays(dataset_dir, sub_folder, samp_rate)
    for sample_array in sample_arrays:
        row = extract_features(sample_array, samp_rate, frame_size, hop_size)
        data = np.array(row)
        fulldata+=[data]
        labels.append(sub_folder)
   


print("Normalizing the data...")
dataset=[]
for i in range(len(fulldata)) :
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    dataset += [scaler.fit_transform(fulldata[i])]

dataset=np.array(dataset)
labels=np.array(labels)

np.save("C:/Users/PAULO/Documents/GitHub/Music-Classifier/mfccdata_fma.npy",dataset)
np.save("C:/Users/PAULO/Documents/GitHub/Music-Classifier/labels_fma.npy",labels)
print("Data set is done !")

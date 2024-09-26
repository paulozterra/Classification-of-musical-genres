import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
def convertir_a_espectrograma(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            audio_path = os.path.join(root, file)

            y, sr = librosa.load(audio_path, sr=None)

            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            output_path = os.path.join(output_folder, file.replace('.wav', '_mel.png'))
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Espectrograma de Mel')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

input_folder_path = 'C:/Users/PAULO/Documents/GitHub/Music-Classifier/FMA_WAVs'
output_folder_path = 'C:/Users/PAULO/Documents/GitHub/Music-Classifier/FMA_MEL'

convertir_a_espectrograma(input_folder_path, output_folder_path)

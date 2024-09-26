import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt


input_folder_path = 'C:/Users/PAULO/Documents/GitHub/Music-Classifier/GTZAN/Data/images_original'
output_folder_path = 'C:/Users/PAULO/Documents/GitHub/Music-Classifier/GTZAN/Data'

def cargar_datos(input_folder):
    datos = []
    etiquetas = []

    for genero in os.listdir(input_folder):
        genero_path = os.path.join(input_folder, genero)

        for archivo in os.listdir(genero_path):
            archivo_path = os.path.join(genero_path, archivo)
            img = image.load_img(archivo_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            datos.append(img_array)
            etiquetas.append(genero)

    datos = np.array(datos)
    etiquetas = np.array(etiquetas)

    return datos, etiquetas

X, y = cargar_datos(input_folder_path)

X, y = cargar_datos(input_folder_path)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape((X_train.shape[0], 128, 128 * 3))
X_test = X_test.reshape((X_test.shape[0], 128, 128 * 3))

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    BatchNormalization(),
    LSTM(64),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

adam_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer= adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history=model.fit(X_train, y_train, epochs=10,verbose=1, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nAccuracy en el conjunto de prueba: {test_acc}')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_pred = model.predict(X_test)
y_pred_binary = np.argmax(y_pred, axis=1)
y_test_binary = np.argmax(y_test, axis=1)

f_score = f1_score(y_test_binary, y_pred_binary, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

print("F-score:", f_score)
print("AUC:", roc_auc)

test_loss, test_acc = model.evaluate(X_test,y_test)
print("test loss :",test_loss)
print("test_accu :",test_acc)
model.save(output_folder_path + '/modelo_lstm_gtzan.h5')

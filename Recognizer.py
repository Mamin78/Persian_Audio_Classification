import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import DataProcessor as dp
import pandas as pd
import numpy as np


def create_model(in_shape):
    model = Sequential()
    model.add(Conv1D(256, kernel_size=6, strides=1, padding='same', activation='relu', input_shape=(in_shape, 1)))
    model.add(AveragePooling1D(pool_size=4, strides=2, padding='same'))

    model.add(Conv1D(128, kernel_size=6, strides=1, padding='same', activation='relu'))
    model.add(AveragePooling1D(pool_size=4, strides=2, padding='same'))

    model.add(Conv1D(128, kernel_size=6, strides=1, padding='same', activation='relu'))
    model.add(AveragePooling1D(pool_size=4, strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, kernel_size=6, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=12, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


class Recognizer:
    def __init__(self, in_shape, X_train=None, y_train=None, X_test=None, y_test=None, checkpoint_filepath='',
                 standard_scaler=None, onehot_encoder=None, n_epochs=75, batch_size=32):
        self.model = create_model(in_shape)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.checkpoint_filepath = checkpoint_filepath
        self.standard_scaler = standard_scaler
        self.onehot_encoder = onehot_encoder

    def set_training_set(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def set_test_set(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def set_n_epochs(self, n_epochs):
        self.n_epochs = n_epochs

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_checkpoint_path(self, checkpoint_filepath):
        self.checkpoint_filepath = checkpoint_filepath

    def set_standard_scaler(self, standard_scaler):
        self.standard_scaler = standard_scaler

    def set_onehot_encoder(self, onehot_encoder):
        self.onehot_encoder = onehot_encoder

    def model_build_summary(self):
        summary = self.model.summary()
        score = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        accuracy = 100 * score[1]
        return summary, accuracy

    def train(self):
        rlrp_callback = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=4, min_lr=0.000001)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_filepath,
                                                                       save_weights_only=True, monitor='val_accuracy',
                                                                       mode='max', save_best_only=True)

        history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.n_epochs,
                                 validation_data=(self.X_test, self.y_test),
                                 callbacks=[rlrp_callback, model_checkpoint_callback])

        return history

    def load_model(self):
        self.model.load_weights(self.checkpoint_filepath)

    def predict_from_extracted_feature(self, features):
        features = self.standard_scaler.transform(np.array([features]))
        features = features.T
        predicted = self.model.predict(np.array([features]))
        return self.onehot_encoder.inverse_transform(predicted)[0][0]

    def predict(self, path_to_file):
        data, sample_rate = dp.load_audio(path_to_file)
        features = dp.extract_features(data)
        return self.predict_from_extracted_feature(features)

    def evaluate(self, features, labels):
        return self.model.evaluate(features, labels, verbose=0)[1]

    def get_confusion_matrix(self, features, labels, saving_path):
        pred_test = self.model.predict(features)
        y_pred = self.onehot_encoder.inverse_transform(pred_test)
        y_test_ = self.onehot_encoder.inverse_transform(labels)

        cm = confusion_matrix(y_test_, y_pred)
        plt.figure(figsize=(12, 10))
        cm = pd.DataFrame(cm, index=[i for i in self.onehot_encoder.categories_],
                          columns=[i for i in self.onehot_encoder.categories_])
        sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
        plt.title('Confusion Matrix for Female Emotions', size=20)
        plt.xlabel('Predicted Labels', size=14)
        plt.ylabel('Actual Labels', size=14)
        plt.savefig(saving_path)

        plt.show()

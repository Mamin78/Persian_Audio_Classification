import matplotlib.pyplot as plt
import argparse
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import Recognizer
import warnings
from joblib import load

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--train', action="store_true",
                    help='Should Model be trained or it has to use pre-trained weights?')
parser.add_argument('--data_frames', action="store_true",
                    help='Are features of sample extracted? -- if not, we should extract them!')
# directories
parser.add_argument('--path_to_data', default='/content/drive/MyDrive/Datasets/12words_cleaned.zip',
                    help='base path to dataset!')

parser.add_argument('--exp_dir', default='experiments', help='path to experiments directory')
parser.add_argument('--checkpoint_filepath', default='/content/exp/01/checkpoints/checkpoint_last.pth', type=str,
                    help='Where the model weights are')

# training
parser.add_argument('--exp', required=True, type=str, help='experiments number e.g. 01')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')

args = parser.parse_args()


def show_graphs(history, saving_path, n_epochs):
    epochs = [i for i in range(n_epochs)]
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    fig.set_size_inches(30, 12)
    ax[0].plot(epochs, train_loss, label='Training Loss')
    ax[0].plot(epochs, test_loss, label='Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs, train_acc, label='Training Accuracy')
    ax[1].plot(epochs, test_acc, label='Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    plt.savefig(saving_path)
    plt.show()


if not os.path.exists(os.path.join(args.exp_dir, args.exp)):
    os.makedirs(os.path.join(args.exp_dir, args.exp))
    os.makedirs(os.path.join(args.exp_dir, args.exp, 'checkpoints'))

x_train = pd.read_csv(os.path.join(args.path_to_data, 'x_train.csv'))
x_test = pd.read_csv(os.path.join(args.path_to_data, 'x_test.csv'))
y_train = pd.read_csv(os.path.join(args.path_to_data, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(args.path_to_data, 'y_test.csv'))

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

encoder = OneHotEncoder()
encoder = load(os.path.join(args.path_to_data, 'encoder.bin'))

scaler = load(os.path.join(args.path_to_data, 'std_scaler.bin'))

recognizer_model = Recognizer.Recognizer(x_train.shape[1], X_train=x_train, y_train=y_train, X_test=x_test,
                                         y_test=y_test,
                                         checkpoint_filepath=args.checkpoint_filepath,
                                         standard_scaler=scaler, onehot_encoder=encoder, n_epochs=args.n_epochs,
                                         batch_size=args.batch_size)
summary, _ = recognizer_model.model_build_summary()
print(summary)

if args.train:
    history = recognizer_model.train()
    show_graphs(history, os.path.join(args.exp_dir, args.exp, 'history.png'), args.n_epochs)
    score_on_train_set = recognizer_model.evaluate(x_train, y_train)
    print('Final Training accuracy: ', score_on_train_set)
else:
    recognizer_model.set_checkpoint_path(args.checkpoint_filepath)
    recognizer_model.load_model()

score_on_test_set = recognizer_model.evaluate(x_test, y_test)
print('Test accuracy: ', score_on_test_set)

recognizer_model.get_confusion_matrix(x_test, y_test, os.path.join(args.exp_dir, args.exp, 'confusion_matrix.png'))

# print("Let's model's performance on some random samples!")
# some_test_samples = metadata.sample(n=5)
# for path, label in zip(some_test_samples.path, some_test_samples.labels):
#     predicted_result = recognizer_model.predict(path)
#     print('The real label is: ', label, ' and the predicted label is: ', predicted_result)

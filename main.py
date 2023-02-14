import matplotlib.pyplot as plt
import argparse
import os
import DataProcessor as dp
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
import Recognizer
import warnings
from tqdm import tqdm
from joblib import dump, load

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--train', action="store_true",
                    help='Should Model be trained or it has to use pre-trained weights?')
parser.add_argument('--data_frames', action="store_true",
                    help='Are features of sample extracted? -- if not, we should extract them!')
# directories
parser.add_argument('--path_to_metadata', default='/content/metadata.csv',
                    help='path to the csv that contains path and label of each sample')
parser.add_argument('--path_to_data', default='/content/drive/MyDrive/Datasets/12words_cleaned.zip',
                    help='base path to dataset!')
parser.add_argument('--path_to_features', default='/content/drive/MyDrive/Datasets/12words_cleaned.zip',
                    help='It is the path to features if they are extracted previously!')
parser.add_argument('--path_to_a_test_file', default='/content/drive/MyDrive/Datasets/12words_cleaned.zip',
                    help='Just a test file')
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


def setup_dataframe(features, labels, path_to_features):
    df = pd.DataFrame(features)
    df['labels'] = labels
    df.to_csv(path_to_features, index=False)
    df.sample(frac=1).head()

    return df


if not os.path.exists(os.path.join(args.exp_dir, args.exp)):
    os.makedirs(os.path.join(args.exp_dir, args.exp))
    os.makedirs(os.path.join(args.exp_dir, args.exp, 'checkpoints'))

metadata = pd.read_csv(args.path_to_metadata)

X, Y = [], []
if not args.data_frames:
    print('Extracting features from audio file:')
    for i, record in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        path, label = record.path, record.labels
        features = dp.get_features(os.path.join(args.path_to_data, path))
        for elem in features:
            X.append(elem)
            Y.append(label)

    print(f'Check shapes:\nFeatures: {len(X)}, Labels: {len(Y)}')

if not args.data_frames:
    Features_data = setup_dataframe(X, Y, args.path_to_features)
else:
    Features_data = pd.read_csv(args.path_to_features, index_col=False)

print(Features_data)
print()
print()
print(len(Features_data.iloc[0]))

X = Features_data.iloc[:, :-1].values
Y = Features_data['labels'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

nogender_X = np.concatenate((X, X))
nogender_Y = np.concatenate((Y, Y))

x_train, x_test, y_train, y_test = train_test_split(nogender_X, nogender_Y, random_state=0, test_size=0.20,
                                                    shuffle=True)
print('After spiting into train and test parts!!')
print("x_train's shape: ", x_train.shape, "y_train's shape: ", y_train.shape, "x_test's shape: ", x_test.shape,
      "y_test's shape: ", y_test.shape)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
# dump(scaler, 'std_scaler.bin', compress=True)
x_test = scaler.transform(x_test)

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
print()
print('After changing dimensions!')
print("x_train's shape: ", x_train.shape, "y_train's shape: ", y_train.shape, "x_test's shape: ", x_test.shape,
      "y_test's shape: ", y_test.shape)

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

print("Let's model's performance on some random samples!")
some_test_samples = metadata.sample(n=5)
for path, label in zip(some_test_samples.path, some_test_samples.labels):
    predicted_result = recognizer_model.predict(path)
    print('The real label is: ', label, ' and the predicted label is: ', predicted_result)

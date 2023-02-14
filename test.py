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
# directories
parser.add_argument('--path_to_metadata', default='/content/metadata.csv',
                    help='path to the csv that contains path and label of each sample')
parser.add_argument('--path_to_data', default='/content/drive/MyDrive/Datasets/12words_cleaned.zip',
                    help='base path to dataset!')

parser.add_argument('--exp_dir', default='experiments', help='path to experiments directory')
parser.add_argument('--checkpoint_filepath', default='/content/exp/01/checkpoints/checkpoint_last.pth', type=str,
                    help='Where the model weights are')
parser.add_argument('--exp', required=True, type=str, help='experiments number e.g. 01')

args = parser.parse_args()

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

encoder = load(os.path.join(args.path_to_data, 'encoder.bin'))
scaler = load(os.path.join(args.path_to_data, 'std_scaler.bin'))

recognizer_model = Recognizer.Recognizer(x_train.shape[1], X_train=x_train, y_train=y_train, X_test=x_test,
                                         y_test=y_test,
                                         checkpoint_filepath=args.checkpoint_filepath,
                                         standard_scaler=scaler, onehot_encoder=encoder)
summary, _ = recognizer_model.model_build_summary()
print(summary)

recognizer_model.set_checkpoint_path(args.checkpoint_filepath)
recognizer_model.load_model()

score_on_test_set = recognizer_model.evaluate(x_test, y_test)
print('Test accuracy: ', score_on_test_set)

recognizer_model.get_confusion_matrix(x_test, y_test, os.path.join(args.exp_dir, args.exp, 'confusion_matrix.png'))

metadata = pd.read_csv(args.path_to_metadata)
some_test_samples = metadata.sample(n=5)
for path, label in zip(some_test_samples.path, some_test_samples.labels):
    predicted_result = recognizer_model.predict(path)
    print('The real label is: ', label, ' and the predicted label is: ', predicted_result)

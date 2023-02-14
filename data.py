import argparse
import os
import DataProcessor as dp
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from joblib import dump


def setup_dataframe(features, labels):
    df = pd.DataFrame(features)
    df['labels'] = labels
    df.sample(frac=1).head()

    return df


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

parser.add_argument('--path_to_metadata', default='/content/metadata.csv',
                    help='path to the csv that contains path and label of each sample')
parser.add_argument('--path_to_data', default='/content/drive/MyDrive/Datasets/12words_cleaned.zip',
                    help='base path to dataset!')
parser.add_argument('--saving_path', default='/content/drive/MyDrive/Datasets/12words_cleaned.zip',
                    help='Anything that is generated in this file will be save in this path')
args = parser.parse_args()

if not os.path.exists(args.saving_path):
    os.makedirs(args.saving_path)

metadata = pd.read_csv(args.path_to_metadata)

X, Y = [], []
print('Extracting features from audio file:')
for i, record in tqdm(metadata.iterrows(), total=metadata.shape[0]):
    path, label = record.path, record.labels
    features = dp.get_features(os.path.join(args.path_to_data, path))
    for elem in features:
        X.append(elem)
        Y.append(label)

print(f'Check shapes:\nFeatures: {len(X)}, Labels: {len(Y)}')

Features_data = setup_dataframe(X, Y)

X = Features_data.iloc[:, :-1].values
Y = Features_data['labels'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
dump(encoder, os.path.join(args.saving_path, 'encoder.bin'), compress=True)

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.20,
                                                    shuffle=True)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
dump(scaler, os.path.join(args.saving_path, 'std_scaler.bin'), compress=True)
x_test = scaler.transform(x_test)

print("Let's see the dimensions!")
print("x_train's shape: ", x_train.shape, "y_train's shape: ", y_train.shape, "x_test's shape: ", x_test.shape,
      "y_test's shape: ", y_test.shape)

pd.DataFrame(x_train).to_csv(os.path.join(args.saving_path, 'x_train.csv'), index=False)
pd.DataFrame(x_test).to_csv(os.path.join(args.saving_path, 'x_test.csv'), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(args.saving_path, 'y_train.csv'), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(args.saving_path, 'y_test.csv'), index=False)

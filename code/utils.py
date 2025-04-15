import numpy as np
import pandas as pd
import random
from sktime.transformations.panel.rocket import Rocket
from sklearn.preprocessing import LabelEncoder

# Better interface with helper functions
class RocketKernel:

    def __init__(self, seed, ts_length, ppv_only=True):
        self.rocket = Rocket(num_kernels=1, random_state=int(seed))
        self.rocket.fit(np.zeros((1, 1, ts_length)))
        self.ppv_only = ppv_only

    def transform(self, X):
        if len(X.shape) <= 1:
            X = X.reshape(1, -1)

        X_transformed = self.rocket.transform(np.expand_dims(X, 1)).to_numpy()
        if self.ppv_only:
            X_transformed = X_transformed[:, 0:1]

        return X_transformed

def transform_seeds(X, seeds, ts_length):
    K = [RocketKernel(int(seed), ts_length, ppv_only=True) for seed in seeds]
    return np.concatenate([k.transform(X) for k in K], axis=1)

def get_binary_dataset_names():
    return [
        'BeetleFly',
        'BirdChicken',
        'Chinatown',
        'Coffee',
        'Computers',
        'DistalPhalanxOutlineCorrect',
        'DodgerLoopGame',
        'DodgerLoopWeekend',
        'Earthquakes',
        'ECG200',
        'ECGFiveDays',
        'FordA',
        'FordB',
        'FreezerRegularTrain',
        'FreezerSmallTrain',
        'GunPoint',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'Ham',
        'HandOutlines',
        'Herring',
        'HouseTwenty',
        'ItalyPowerDemand',
        'Lightning2',
        'MiddlePhalanxOutlineCorrect',
        'MoteStrain',
        'PhalangesOutlinesCorrect',
        'PowerCons',
        'ProximalPhalanxOutlineCorrect',
        'SemgHandGenderCh2',
        'ShapeletSim',
        'SonyAIBORobotSurface1',
        'SonyAIBORobotSurface2',
        'Strawberry',
        'ToeSegmentation1',
        'ToeSegmentation2',
        'TwoLeadECG',
        'Wafer',
        'Wine',
        'WormsTwoClass',
        'Yoga'
    ]

def get_three_classes_dataset_names():
    return [
        'ArrowHead',
        'CBF',
        'ChlorineConcentration',
        'DistalPhalanxOutlineAgeGroup',
        'LargeKitchenAppliances',
        'Meat',
        'MiddlePhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineAgeGroup',
        'RefrigerationDevices',
        'ScreenType',
        'SmallKitchenAppliances',
        'StarLightCurves',
        'BME',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'SmoothSubspace',
        'UMD'
    ]

def get_four_classes_dataset_names():
    return [
        'Car',
        'CinCECGTorso',
        'DiatomSizeReduction',
        'FaceFour',
        'OliveOil',
        'Trace',
        'TwoPatterns',
        'EthanolLevel',
        'Rock'
    ]

def get_multiclasses_dataset_names():
    return [
        'Beef', 'ECG5000', 'Haptics', 'Worms', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'SemgHandSubjectCh2', 'DistalPhalanxTW', 'MiddlePhalanxTW', 'OSULeaf', 'ProximalPhalanxTW', 'Symbols', 'SyntheticControl', 'GesturePebbleZ1', 'GesturePebbleZ2', 'SemgHandMovementCh2', 'ElectricDevices', 'Fish', 'InlineSkate', 'Lightning7', 'Plane', 'DodgerLoopDay', 'Mallat', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'MedicalImages', 'ACSF1', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'MelbournePedestrian', 'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ', 'InsectWingbeatSound', 'PLAID', 'CricketX', 'CricketY', 'CricketZ', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'FaceAll', 'FacesUCR', 'SwedishLeaf', 'Fungi', 'Crop', 'WordSynonyms', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'Adiac', 'Phoneme', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'FiftyWords', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'ShapesAll']


def load_dataset(ds_name):
    LE = LabelEncoder()
    ds_train = pd.read_table(f'../UCRArchive_2018/{ds_name}/{ds_name}_TRAIN.tsv', header=None).to_numpy()
    X_train = ds_train[:, 1:]
    Y_train = ds_train[:, 0]
    Y_train = LE.fit_transform(Y_train)
    ds_test = pd.read_table(f'../UCRArchive_2018/{ds_name}/{ds_name}_TEST.tsv', header=None).to_numpy()
    X_test = ds_test[:, 1:]
    Y_test = ds_test[:, 0]
    Y_test = LE.transform(Y_test)
    return X_train, Y_train, X_test, Y_test

def preprocess_data(X_train, Y_train, X_test, Y_test):
    # TODO: Scale each time series?
    # TODO: NaN values? Should be none in the data
    # TODO: LabelEncoder necessary if we do not use PyTorch?
    return X_train, Y_train, X_test, Y_test

# Split data continuously
def split_data_continuous(features, target, num_clients):
    client_data = []
    data_per_client = len(features) // num_clients

    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client
        client_features = features[start_idx:end_idx]
        client_target = target[start_idx:end_idx]
        client_data.append((client_features, client_target))

    return client_data

# Uniform split data with equal distribution
def split_data_uniform(features, target, num_clients, rng):
    # Check if number of clients is valid
    if num_clients <= 0:
        raise ValueError("Number of clients must be greater than 0.")
    unique_labels = np.unique(target)

    X_clients = [[] for _ in range(num_clients)]
    Y_clients = [[] for _ in range(num_clients)]
    
    shuffled_indices = rng.permutation(len(features))

    # Distribute one sample of each class to each client (if available)
    for class_label in unique_labels:
        class_indices = np.where(target == class_label)[0]
        rng.shuffle(class_indices)
        # Only distribute up to min(num_clients, number of samples for this class)
        num_to_distribute = min(num_clients, len(class_indices))
        for i in range(num_to_distribute):
            index = class_indices[i]
            X_clients[i].append(features[index])
            Y_clients[i].append(target[index])

    # Remaining samples in uniform distribution
    for index in shuffled_indices:
        x_sample, y_sample = features[index], target[index]
        # Skip if already distributed in the first phase
        already_distributed = any(
            np.array_equal(x_sample, x) and y_sample == y
            for client_x, client_y in zip(X_clients, Y_clients)
            for x, y in zip(client_x, client_y)
        )
        if not already_distributed:
            min_samples_client = min(range(num_clients), key=lambda k: len(Y_clients[k]))
            X_clients[min_samples_client].append(x_sample)
            Y_clients[min_samples_client].append(y_sample)

    # Convert lists to numpy arrays
    X_clients = [np.array(client) for client in X_clients]
    Y_clients = [np.array(client) for client in Y_clients]

    return X_clients, Y_clients

# Random split data
def split_data_random(features, target, num_clients):
    combined_df = pd.concat([features, target], axis=1)
    
    combined_df = combined_df.sample(frac=1, random_state=myseed).reset_index(drop=True)
    
    client_data = []
    data_per_client = len(combined_df) // num_clients

    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client
        client_df = combined_df[start_idx:end_idx]
        client_features = client_df.iloc[:, :-1]
        client_target = client_df.iloc[:, -1]
        client_data.append((client_features, client_target))

    return client_data

def split_function(features, target, num_clients, random_state, my_split='uniform'):
    if my_split == 'continuous':
        return split_data_continuous(features, target, num_clients)
    elif my_split == 'uniform':
        return split_data_uniform(features, target, num_clients, random_state)
    elif my_split == 'random':
        return split_data_random(features, target, num_clients)
    else:
        raise ValueError("Invalid split type. Choose 'continuous', 'uniform', or 'random'.")

import numpy as np
import pandas as pd
import random
import functools

from functools import lru_cache
from sklearn.linear_model import RidgeClassifier
from sktime.transformations.panel.rocket import Rocket, MiniRocket
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split

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

def load_dataset(ds_name):
    LE = LabelEncoder()
    ds_train = pd.read_table(f'../../UCRArchive_2018/{ds_name}/{ds_name}_TRAIN.tsv', header=None).to_numpy()
    X_train = ds_train[:, 1:]
    Y_train = ds_train[:, 0]
    Y_train = LE.fit_transform(Y_train)
    ds_test = pd.read_table(f'../../UCRArchive_2018/{ds_name}/{ds_name}_TEST.tsv', header=None).to_numpy()
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

def split_data_dirichlet(train_features, train_targets, test_features, test_targets, num_clients, alpha=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    num_classes = len(set(train_targets))
    client_data = {i: {'train': {'features': [], 'targets': []},
                      'test': {'features': [], 'targets': []}} for i in range(num_clients)}
    
    # Check if we have enough samples per class
    min_samples_per_class = num_clients  # Need at least one per client
    train_class_counts = np.bincount(train_targets)
    test_class_counts = np.bincount(test_targets)
    
    for c in range(num_classes):
        if train_class_counts[c] < min_samples_per_class:
            raise ValueError(f"Class {c} has only {train_class_counts[c]} train samples but needs at least {min_samples_per_class}")
        if test_class_counts[c] < min_samples_per_class:
            raise ValueError(f"Class {c} has only {test_class_counts[c]} test samples but needs at least {min_samples_per_class}")
    
    def get_class_indices(features, targets):
        return {i: np.where(np.array(targets) == i)[0] for i in range(num_classes)}
    
    train_class_indices = get_class_indices(train_features, train_targets)
    test_class_indices = get_class_indices(test_features, test_targets)

    class_proportions = {c: np.random.dirichlet(alpha * np.ones(num_clients)) for c in range(num_classes)}

    def assign_data(features, targets, class_indices, split_name):
        for c in range(num_classes):
            indices = class_indices[c]
            np.random.shuffle(indices)
            
            mandatory_assignments = indices[:num_clients]
            remaining_indices = indices[num_clients:]
            
            for i in range(num_clients):
                idx = mandatory_assignments[i]
                client_data[i][split_name]['features'].append(features[idx])
                client_data[i][split_name]['targets'].append(targets[idx])
            
            if len(remaining_indices) > 0:
                proportions = (np.cumsum(class_proportions[c]) * len(remaining_indices)).astype(int)[:-1]
                client_splits = np.split(remaining_indices, proportions)
                
                for i, split in enumerate(client_splits):
                    client_data[i][split_name]['features'].extend(features[idx] for idx in split)
                    client_data[i][split_name]['targets'].extend(targets[idx] for idx in split)
    
    assign_data(train_features, train_targets, train_class_indices, 'train')
    assign_data(test_features, test_targets, test_class_indices, 'test')
    
    for i in range(num_clients):
        for split in ['train', 'test']:
            client_data[i][split]['features'] = np.array(client_data[i][split]['features'])
            client_data[i][split]['targets'] = np.array(client_data[i][split]['targets'])
    
    return ([client_data[i]['train']['features'] for i in range(num_clients)],
            [client_data[i]['train']['targets'] for i in range(num_clients)],
            [client_data[i]['test']['features'] for i in range(num_clients)],
            [client_data[i]['test']['targets'] for i in range(num_clients)])

def split_function(features, target, num_clients, random_state, my_split='uniform'):
    if my_split == 'continuous':
        return split_data_continuous(features, target, num_clients)
    elif my_split == 'uniform':
        return split_data_uniform(features, target, num_clients, random_state)
    elif my_split == 'random':
        return split_data_random(features, target, num_clients)
    else:
        raise ValueError("Invalid split type. Choose 'continuous', 'uniform', or 'random'.")

ds_name="ArrowHead"
n_clients = 4
n_rounds = 100
n_kernels = 84 #924 #9996
lambda_reg = 0.01
n_important_weights = (n_kernels * 1) // n_clients
top_k_per_client = (n_kernels * 1) // n_clients
rng = np.random.RandomState(1)
np.random.seed(1)
random.seed(1)
myseed = 1

#ds_name = "BeetleFly"
print('DATASET', ds_name)

X_train, Y_train, X_test, Y_test = load_dataset(ds_name)

n_classes = len(np.unique(np.concatenate([Y_train, Y_test], axis=0)))
X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)

#UNIFORM
s_X_train, s_Y_train = split_function(X_train, Y_train, n_clients, rng)
s_X_test, s_Y_test = split_function(X_test, Y_test, n_clients, rng)

#DIRICHLET 0.5
#s_X_train, s_Y_train, s_X_test, s_Y_test = split_data_dirichlet(X_train, Y_train, X_test, Y_test, n_clients, 0.1, myseed)

minirocket = MiniRocket(num_kernels=n_kernels, random_state=int(myseed))
#minirocket.fit(np.expand_dims(X_train, 1)) #vogliamo fittare non qui sul dataset aggregato ma poi sui singoli clients, per avere diversi bias

ts_length = len(X_train[0])
n_classes = len(np.unique(Y_train))

for epoch in range(n_rounds +1):
    if epoch == 1:
        break
    local_biases = []
    train_accs = np.array([], dtype=float)
    round_selected_seeds = np.array([], dtype=int)
    train_accuracies = np.array([], dtype=float)
    lambdas = np.array([], dtype=float)
    only_lambdas = np.array([], dtype=float)
    for client_id in range(n_clients):
        x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
        #print('lunghezza x ', x_train)
        #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)
        minirocket.fit(np.expand_dims(x_train, 1)) #fittiamo qui in quanto ogni client avrà dati diversi e quindi bias diversi
        x_trans = minirocket.transform(np.expand_dims(x_train, 1)).to_numpy()
        #print(minirocket.parameters[2])
        print(minirocket.parameters[2][:5])
        local_biases.append(minirocket.parameters[2].copy())
        model = RidgeClassifierCV(alphas=np.logspace(-3,0,100)).fit(x_trans, y_train)

        train_acc = model.score(x_trans, y_train)
        train_accuracies = np.concatenate([train_accuracies, [train_acc]])
        lambdas = np.concatenate([lambdas, [model.alpha_ * train_acc]])
        only_lambdas = np.concatenate([only_lambdas, [model.alpha_]])
        train_accs = np.concatenate([train_accs, [train_acc]])
        
    mean_train_accuracy = np.mean(train_accuracies)

    global_biases = np.stack(local_biases)
    mean_biases = np.mean(global_biases, axis=0) 

    print(f"\n=== Round {epoch + 1} ===, Mean Train Accuracy: {mean_train_accuracy:.4f}, Number of unique seeds: {n_kernels}")
            
A_global = np.zeros((n_kernels, n_kernels))
b_global = np.zeros((n_kernels, n_classes))
A_locals, b_locals = [], []

new_parameters = (minirocket.parameters[0], minirocket.parameters[1], mean_biases.copy())
minirocket.parameters = new_parameters
print(minirocket.parameters[2][:5])

for client_id in range(n_clients):
    x_train, y_train = s_X_train[client_id], s_Y_train[client_id]    
    x_trans = minirocket.transform(np.expand_dims(x_train, 1)).to_numpy()
    y_onehot = np.eye(n_classes)[y_train]
    A_local = x_trans.T @ x_trans
    b_local = x_trans.T @ y_onehot
    A_locals.append(A_local)
    b_locals.append(b_local)
            
    A_global += A_local
    b_global += b_local

#print("b global shape:", b_global.shape)

#cross validation for best lambda
local_best_lambdas = np.array([])
local_best_accs = np.array([])
tmp_lambda = 0
for client_id in range(n_clients):
    best_accuracy = 0
    x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
    #_, x_val, _, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)
    x_trans = minirocket.transform(np.expand_dims(x_train, 1)).to_numpy() 
    y_onehot = np.eye(n_classes)[y_train]

    #ROBERTATA
    A_minus_client = A_global - A_locals[client_id]
    b_minus_client = b_global - b_locals[client_id]
    
    for cv_lambda in np.logspace(-3,0,100):
        local_A_final = A_minus_client + cv_lambda * np.eye(A_global.shape[0]) #ROBERTATA CON A_MINUS, prima era A_global
        local_W_final = np.linalg.solve(local_A_final, b_minus_client) #ROBERTATA prima era b_global
        local_W_final =  local_W_final / (np.linalg.norm(local_W_final, axis=0, keepdims=True) + 1e-10)
        local_y_pred = np.argmax(x_trans @ local_W_final, axis=1)
        local_accuracy = accuracy_score(y_train, local_y_pred)
        if best_accuracy < local_accuracy or (best_accuracy == local_accuracy and cv_lambda > tmp_lambda):
            tmp_lambda = cv_lambda
            best_accuracy = local_accuracy
        #print("Client: ", client_id, "best local accuracy: ", local_accuracy, "with lambda: ", cv_lambda)
    local_best_lambdas = np.concatenate([local_best_lambdas, [tmp_lambda * best_accuracy]])
    local_best_accs = np.concatenate([local_best_accs, [best_accuracy]])

print(f"\n--- FED2RC --- Lambdas: {local_best_lambdas}", f"Mean lambda: {np.sum(local_best_lambdas)/np.sum(local_best_accs)}") 

A_final = A_global + np.sum(local_best_lambdas)/np.sum(local_best_accs) * np.eye(A_global.shape[0])
W_final = np.linalg.solve(A_final, b_global)
W_final =  W_final / (np.linalg.norm(W_final, axis=0, keepdims=True) + 1e-10)
X_test_trans = minirocket.transform(np.expand_dims(X_test, 1)).to_numpy()
y_pred = np.argmax(X_test_trans @ W_final, axis=1)
accuracy = accuracy_score(Y_test, y_pred)
if n_classes == 2:
    test_f1 = f1_score(Y_test, y_pred)
else:
    test_f1 = f1_score(Y_test, y_pred, average = 'macro')

print(f"\n--- FED2RC --- Test Accuracy: {accuracy:.4f}, Test F1: {test_f1}, Number of unique seeds: {n_kernels}")

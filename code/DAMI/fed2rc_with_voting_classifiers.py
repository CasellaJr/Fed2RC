import numpy as np
import pandas as pd
import random
import wandb
import argparse
import itertools
import os

from sklearn.linear_model import RidgeClassifier
from sktime.transformations.panel.rocket import Rocket
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from utils import load_dataset, preprocess_data, split_function, RocketKernel, transform_seeds, get_binary_dataset_names, get_three_classes_dataset_names, get_four_classes_dataset_names, get_multiclasses_dataset_names
from sklearn.preprocessing import LabelBinarizer

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--clients', type=int, default=4, help='Choose the number of parties of the federation')
parser.add_argument('-k', '--kernels', type=int, default=100, help='Choose the number of ROCKET kernels to use')
parser.add_argument('-r', '--rounds', type=int, default=100, help='Choose the number of rounds of federated training')
parser.add_argument('-v', '--validation', action='store_true', help='Choose this option to enable a grid search for the lambda regularizer on a validation set.')
parser.add_argument('--debug', action='store_true', help='Use this option to disable WandB metrics tracking')
args = parser.parse_args()

# Constants
n_clients = args.clients 
n_kernels = args.kernels
n_rounds = args.rounds
top_k_per_client = (n_kernels * 1) // n_clients
EXPERIMENT_SEEDS = [1,2,3,4,5]

def main():
    #list_of_datasets = get_binary_dataset_names()
    #list_of_datasets = get_three_classes_dataset_names()
    #list_of_datasets = get_four_classes_dataset_names()
    #list_of_datasets = get_multiclasses_dataset_names()
    #list_of_datasets = ['PigCVP'] #example
    list_of_datasets = get_binary_dataset_names() + get_three_classes_dataset_names() + get_four_classes_dataset_names() + get_multiclasses_dataset_names()

    if args.debug:
        wandb.init(mode="disabled")
    else:
        None

    # All the runs
    run_names = []
    runs = wandb.Api().runs("mlgroup/Fed2RC")
    for run in runs:
        run_names.append(run.name)
    #print("RUN NAMES DONE: ", run_names)
    wandb.finish()

    for ds_name in list_of_datasets:
        print('DATASET', ds_name)

        X_train, Y_train, X_test, Y_test = load_dataset(ds_name)

        n_classes = len(np.unique(np.concatenate([Y_train, Y_test], axis=0)))
        X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)

        if args.debug:
            wandb.init(mode="disabled")
        else:
            entity = "mlgroup"
            project = "Fed2RC"
            if args.validation:
                run_name = f"VAL_{ds_name}_{n_kernels}KERNELS_{n_clients}CLIENTS"
            else:
                run_name = f"{ds_name}_{n_kernels}KERNELS_{n_clients}CLIENTS"
            tags = ["Fed2RC"]
            if run_name in run_names:
                print(f"Experiment {run_name} already executed.")
                continue
            else:
                wandb.init(project=project, entity=entity, group=f"{run_name}", name=run_name, tags=tags)

        for experiment_seed in EXPERIMENT_SEEDS:
            print('SEED', experiment_seed)
            rng = np.random.RandomState(experiment_seed)
            np.random.seed(experiment_seed)
            random.seed(experiment_seed)
            os.environ['PYTHONHASHSEED'] = str(experiment_seed)
            
            if ds_name in ["DiatomSizeReduction", "Fungi", "PigAirwayPressure", "PigArtPressure", "PigCVP"]:
                split = "random"
                s_X_Y_train = split_function(X_train, Y_train, n_clients, rng, my_split=split)
                s_X_Y_test = split_function(X_test, Y_test, n_clients, rng, my_split=split)
                s_X_train = [x for x, _ in s_X_Y_train]
                s_Y_train = [y for _, y in s_X_Y_train]
                s_X_test = [x for x, _ in s_X_Y_test]
                s_Y_test = [y for _, y in s_X_Y_test]
            else:
                split='uniform'
                s_X_train, s_Y_train = split_function(X_train, Y_train, n_clients, rng, my_split=split)
                s_X_test, s_Y_test = split_function(X_test, Y_test, n_clients, rng, my_split=split)
            

            # Initialize different seeds for all clients
            seeds = np.arange(n_clients * n_kernels).reshape(n_clients, n_kernels)

            dict_seeds = {}
            for client_id in range(n_clients):
                dict_seeds[client_id] = seeds[client_id]

            ts_length = len(X_train[0])
            n_classes = len(np.unique(Y_train))
            kernels = [[RocketKernel(seed=int(seed), ts_length=ts_length, ppv_only=True) for seed in dict_seeds[client_id]] for client_id in range(n_clients)]


            best_accuracy = 0
            best_seeds = None
            previous_seeds = 0
            consecutive_same_seeds = 0 
            selected_seeds = {}

            for epoch in range(n_rounds +1):
                train_accs = np.array([], dtype=float)
                round_selected_seeds = np.array([], dtype=int)
                train_accuracies = np.array([], dtype=float)
                lambdas = np.array([], dtype=float)
                for client_id in range(n_clients):
                    x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
                    if args.validation:
                        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=experiment_seed)
                    client_seeds = dict_seeds[client_id]
                    x_trans = transform_seeds(x_train, client_seeds, ts_length)
                    model = RidgeClassifierCV(alphas=np.logspace(-3,0,100)).fit(x_trans, y_train)
                    if model.coef_.shape[0] == 1:  # Binary classification: extract kernels associated to largest squared weights
                        w = model.coef_[0]
                        top_k_idx = np.argsort(-np.abs(w))[:top_k_per_client]
                    else:  # Multiclass classification: softmax over the weights to extract kernels as voting. 
                        votes = np.zeros((n_classes,n_kernels), dtype=float)
                        for class_idx in range(n_classes):
                            w_class = np.abs(model.coef_[class_idx])
                            w_class_normalized = np.exp(w_class)/sum(np.exp(w_class))
                            votes[class_idx] = w_class_normalized
                        kernel_votes = np.mean(votes, axis=0)
                        sorted_kernel_votes = np.argsort(-kernel_votes)
                        top_k_idx = sorted_kernel_votes[:top_k_per_client]
                    top_k_seeds = client_seeds[top_k_idx]
                    round_selected_seeds = np.concatenate([round_selected_seeds, top_k_seeds])
                    lambdas = np.concatenate([lambdas, [model.alpha_ * train_acc]])
                    train_accs = np.concatenate([train_accs, [train_acc]])
                    
                round_selected_seeds = np.unique(round_selected_seeds)
                final_num_kernels = len(round_selected_seeds)
                mean_train_accuracy = np.mean(train_accuracies)
                    
                print(f"\n=== Round {epoch + 1} ===, Mean Train Accuracy: {mean_train_accuracy:.4f}, Number of unique seeds: {final_num_kernels}")
                    
                if previous_seeds is not None and np.array_equal(previous_seeds, round_selected_seeds):
                    consecutive_same_seeds += 1
                    if consecutive_same_seeds >= 1: 
                        print("\n Convergence achieved: Seeds unchanged for 2 consecutive epochs. Training terminated.")
                        break
                else:
                    consecutive_same_seeds = 0
                
                previous_seeds = round_selected_seeds.copy()
                
                for client_id in range(n_clients):
                    dict_seeds[client_id] = round_selected_seeds.copy()
                        
                    
            A_global = np.zeros((final_num_kernels, final_num_kernels))
            b_global = np.zeros((final_num_kernels, n_classes))
            A_locals, b_locals = [], []
            for client_id in range(n_clients):
                x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
                client_seeds = dict_seeds[client_id]
                x_trans = transform_seeds(x_train, client_seeds, ts_length)  
                y_onehot = np.eye(n_classes)[y_train]
                A_local = x_trans.T @ x_trans
                b_local = x_trans.T @ y_onehot
                A_locals.append(A_local)
                b_locals.append(b_local)
                        
                A_global += A_local
                b_global += b_local


            #cross validation for best lambda
            local_best_lambdas = np.array([])
            local_best_accs = np.array([])
            tmp_lambda = 0
            for client_id in range(n_clients):
                best_accuracy = 0
                x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
                if args.validation:
                    _, x_val, _, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=experiment_seed)
                client_seeds = dict_seeds[client_id]
                x_trans = transform_seeds(x_train, client_seeds, ts_length)  
                if args.validation:
                    x_trans = transform_seeds(x_val, client_seeds, ts_length)
                y_onehot = np.eye(n_classes)[y_train]

                A_minus_client = A_global - A_locals[client_id]
                b_minus_client = b_global - b_locals[client_id]
                
                for cv_lambda in np.logspace(-3,0,100):
                    if args.validation == False:
                        local_A_final = A_minus_client + cv_lambda * np.eye(A_global.shape[0])
                        local_W_final = np.linalg.solve(local_A_final, b_minus_client)
                    else:
                        local_A_final = A_global + cv_lambda * np.eye(A_global.shape[0])
                        local_W_final = np.linalg.solve(local_A_final, b_global)
                    local_W_final =  local_W_final / (np.linalg.norm(local_W_final, axis=0, keepdims=True) + 1e-10)
                    local_y_pred = np.argmax(x_trans @ local_W_final, axis=1)
                    if args.validation == False:
                        local_accuracy = accuracy_score(y_train, local_y_pred)
                    else:
                        local_accuracy = accuracy_score(y_val, local_y_pred)
                    if best_accuracy < local_accuracy or (best_accuracy == local_accuracy and cv_lambda > tmp_lambda):
                        tmp_lambda = cv_lambda
                        best_accuracy = local_accuracy
                    #print("Client: ", client_id, "best local accuracy: ", local_accuracy, "with lambda: ", cv_lambda)
                local_best_lambdas = np.concatenate([local_best_lambdas, [tmp_lambda * best_accuracy]])
                local_best_accs = np.concatenate([local_best_accs, [best_accuracy]])

            A_final = A_global + np.sum(local_best_lambdas)/np.sum(local_best_accs) * np.eye(A_global.shape[0])
            W_final = np.linalg.solve(A_final, b_global)
            W_final =  W_final / (np.linalg.norm(W_final, axis=0, keepdims=True) + 1e-10)
            X_test_trans = transform_seeds(X_test, round_selected_seeds, ts_length)
            y_pred = np.argmax(X_test_trans @ W_final, axis=1)
            test_accuracy = accuracy_score(Y_test, y_pred)
            if n_classes == 2:
                test_f1 = f1_score(Y_test, y_pred)
            else:
                test_f1 = f1_score(Y_test, y_pred, average = 'macro')

            print(f"Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1}, Number of unique seeds: {n_kernels}")

            if args.validation:    
                wandb.log({f'Test Accuracy using a regularizer coming from a validation set ({experiment_seed})': test_accuracy,
                            f'Test F1 using a regularizer coming from a validation set ({experiment_seed})': test_f1,
                            f'Validation Regularizer': np.sum(local_best_lambdas)/np.sum(local_best_accs),
                            f'Rounds': ({epoch}),
                            f'Kernels': final_num_kernels})
            else:
                wandb.log({f'Test Accuracy ({experiment_seed})': test_accuracy,
                            f'Test F1 ({experiment_seed})': test_f1,
                            f'Validation Regularizer': np.sum(local_best_lambdas)/np.sum(local_best_accs),
                            f'Rounds': ({epoch}),
                            f'Kernels': final_num_kernels})
        wandb.finish()

if __name__ == '__main__':
    main()
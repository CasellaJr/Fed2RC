
from typing import Any, Generator, Iterable, Union

import numpy as np
import torch
import warnings
from fluke import FlukeENV  # NOQA
from fluke.algorithms import CentralizedFL  # NOQA
from fluke.client import Client  # NOQA
from fluke.comm import Message  # NOQA
from fluke.data import FastDataLoader  # NOQA
from fluke.server import EarlyStopping, Server  # NOQA
from fluke.config import OptimizerConfigurator  # NOQA
from rich.progress import track
from sklearn.linear_model import RidgeClassifierCV  # NOQA
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.rocket import Rocket, MiniRocket


class RidgeTorchModel(torch.nn.Module):
    def __init__(self,
                 W: np.ndarray,
                 b: np.ndarray,
                 biases: Iterable[int]):
        super().__init__()
        self.model = torch.nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
        self.biases = biases
        self.minirocket = MiniRocket(num_kernels=len(biases), random_state=FlukeENV().get_seed())
        
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor(W, dtype=torch.float32))
            if b is not None:
                self.model.bias.copy_(torch.tensor(b, dtype=torch.float32))

    def forward(self, X: torch.Tensor):
        self.minirocket.fit(np.expand_dims(X.numpy(), 1))
        new_parameters = (self.minirocket.parameters[0], self.minirocket.parameters[1], self.biases.copy())
        self.minirocket.parameters = new_parameters
        X = torch.tensor(self.minirocket.transform(np.expand_dims(X, 1)).to_numpy(), dtype=torch.float32)
        output = self.model(X)
        return output


def compress_model(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    eigvals, eigvecs = np.linalg.eigh(A)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    U_k = eigvecs[:, :k]
    Lambda_k = np.diag(eigvals[:k])
    return (U_k, Lambda_k)


class ClientFed2RC(Client):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,  # Not used
                 loss_fn: torch.nn.Module,  # Not used
                 local_epochs: int = 3,  # Not used
                 fine_tuning_epochs: int = 0,  # Not used
                 clipping: float = 0,  # Not used
                 n_kernels: int = 1,
                 top_k: int = 1,
                 compression_factor: int = 0,
                 **kwargs: dict[str, Any]):
        assert compression_factor >= 1, "Compression factor must be greater or equal than 1"
        assert n_kernels > 0, "Number of kernels must be greater than 0"
        super().__init__(index, train_set, test_set, None, None, 0, 0, 0, **kwargs)
        self.hyper_params.update(n_kernels=n_kernels, top_k=top_k,
                                 compression_factor=compression_factor)
        self.minirocket = MiniRocket(num_kernels=self.hyper_params.n_kernels, random_state=FlukeENV().get_seed())
        self.local_biases: Iterable[int] = None
        self.converged: bool = False
        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def send_model(self):
        if self.converged:
            self.channel.send(Message((self.A, self.b), "Ab", self.index), "server")
        else:
            self.channel.send(Message(self.local_biases, "biases", self.index), "server")

    def receive_model(self):
        self.converged = self.channel.receive(self.index, "server", msg_type="converge").payload
        if not self.converged:
            msg = self.channel.receive(self.index, "server", msg_type="biases")
            self.local_biases = msg.payload if self._last_round != 1 else self.local_biases

    def fit(self, **kwargs: dict[str, Any]) -> float:
        X, y = self.train_set.tensors
        X, y = X.numpy(), y.numpy()
        #print('lunghezza x', X)
        self.minirocket.fit(np.expand_dims(X, 1))
        if self.local_biases is None:
            self.local_biases = self.minirocket.parameters[2]
        self.minirocket.parameters = (self.minirocket.parameters[0],
                                          self.minirocket.parameters[1],
                                          self.local_biases.copy())
        X_trans = self.minirocket.transform(np.expand_dims(X, 1)).to_numpy()
        self.local_biases = self.minirocket.parameters[2]

        if self.converged:
            y_onehot = np.eye(self.train_set.num_labels)[y]
            self.A = X_trans.T @ X_trans
            self.b = X_trans.T @ y_onehot

            if self.hyper_params.compression_factor > 1:
                k = max(1, self.hyper_params.n_kernels // self.hyper_params.compression_factor)
                if self.hyper_params.compression_factor > self.hyper_params.n_kernels:
                    warnings.warn("Compression factor is greater than the number of kernels. "
                                  "Using the number of kernels instead.")

                U_k, Lambda_k = compress_model(self.A, k)
                self.A = (U_k, Lambda_k)
        else:
            ridge = RidgeClassifierCV(alphas=np.logspace(-3, 0, 100)).fit(X_trans, y)
            #print(ridge.coef_.shape)
            W = np.atleast_2d(ridge.coef_)
            # Useful if client evaluation is needed
            
            self.model = RidgeTorchModel(W,
                                         ridge.intercept_,
                                         self.local_biases)

        return 0.0

    def tune_lambda(self):
        X, y = self.train_set.tensors
        X, y = X.numpy(), y.numpy()
        X_trans = self.minirocket.transform(np.expand_dims(X, 1)).to_numpy()

        A_global, b_global = self.channel.receive(self.index, "server", "Ab").payload
        if isinstance(A_global, tuple):
            U, L = A_global
            A_global = U @ L @ U.T
            self.A = X_trans.T @ X_trans
        A_minus_client = A_global - self.A
        b_minus_client = b_global - self.b

        best_accuracy = 0
        tmp_lambda = 0

        for cv_lambda in np.logspace(-3, 0, 100):
            local_A_final = A_minus_client + cv_lambda * np.eye(A_global.shape[0])
            local_W_final = np.linalg.solve(local_A_final, b_minus_client)
            local_W_final = local_W_final / \
                (np.linalg.norm(local_W_final, axis=0, keepdims=True) + 1e-10)
            local_y_pred = np.argmax(X_trans @ local_W_final, axis=1)
            local_accuracy = accuracy_score(y, local_y_pred)
            if best_accuracy < local_accuracy or \
                    (best_accuracy == local_accuracy and cv_lambda > tmp_lambda):
                tmp_lambda = cv_lambda
                best_accuracy = local_accuracy

        self.channel.send(Message((tmp_lambda * best_accuracy, best_accuracy),
                          "lambda", self.index), "server")


class ServerFed2RC(Server):

    def __init__(self,
                 model: RidgeTorchModel,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = False,
                 lr: float = 1.0,
                 tune_lambda: bool = False,
                 **kwargs: dict[str, Any]):
        super().__init__(model=None, test_set=test_set, clients=clients, **kwargs)
        self.hyper_params.update(tune_lambda=tune_lambda)
        self.global_biases = []
        self.converged = False
        self.compression_k = 0
        self.eligible_clients = set()

    def receive_client_models(self,
                              eligible: Iterable[Client],
                              **kwargs: dict[str, Any]) -> Generator[Any, None, None]:
        if not self.converged:
            for client in eligible:
                client_biases = self.channel.receive("server", client.index, "biases").payload
                yield client_biases
        else:
            for client in eligible:
                client_Ab = self.channel.receive("server", client.index, "Ab").payload
                yield client_Ab

    def broadcast_model(self, eligible: Iterable[Client]) -> None:
        eligible_clients = [client.index for client in eligible]
        self.channel.broadcast(Message(self.converged, "converge", "server"), eligible_clients)
        if not self.converged:
            self.channel.broadcast(Message(self.global_biases, "biases", "server"), eligible_clients)

    def _compute_final_model(self, lam: float = 0.01) -> None:
        A_final = self.A + lam * np.eye(self.A.shape[0])
        W_final = np.linalg.solve(A_final, self.b)
        W_final = W_final / (np.linalg.norm(W_final, axis=0, keepdims=True) + 1e-10)
        self.model = RidgeTorchModel(W_final.T, None, self.global_biases)

    def aggregate(self,
                  eligible: Iterable[Client],
                  client_biases_ab: Generator[Any, None, None]) -> None:
        self.eligible_clients = set(client.index for client in eligible)
        if not self.converged:
            client_biases = list(client_biases_ab)
            #print("Client biases received:", client_biases)
            global_biases = np.stack(client_biases)
            global_biases = np.mean(global_biases, axis=0)
            #print("Global biases:", global_biases, global_biases.shape)
            self.converged = self.rounds == 1
            if not self.converged:
                #print("Not converged yet, waiting for one more round...")
                self.global_biases = global_biases
            self.notify("track_item", round=self.rounds+1, item="biases", value=len(self.global_biases))
        else:
            #print("Global biases:", self.global_biases, self.global_biases.shape)
            A_global = np.zeros((self.global_biases.shape[0], self.global_biases.shape[0]))
            b_global = None

            for client_Ab in client_biases_ab:
                A, b = client_Ab
                if isinstance(A, tuple):
                    U, L = A
                    self.compression_k = U.shape[1]
                    A = U @ L @ U.T
                A_global += A
                if b_global is None:
                    b_global = b
                else:
                    b_global += b
            self.A, self.b = A_global, b_global

            if not self.hyper_params.tune_lambda:
                self._compute_final_model(0.01)

            raise EarlyStopping(self.rounds)

    def finalize(self) -> None:

        if self.converged and self.hyper_params.tune_lambda:
            client_ids = [client.index for client in self.clients]
            if self.compression_k > 0:
                U, L = compress_model(self.A, self.compression_k)
                self.channel.broadcast(Message(((U, L), self.b), "Ab", "server"), client_ids)
            else:
                self.channel.broadcast(Message((self.A, self.b), "Ab", "server"), client_ids)
            local_best_lambdas = np.zeros(len(self.clients))
            local_best_accs = np.zeros(len(self.clients))
            for client in track(self.clients, description="Validating lambda...", transient=True):
                if client.index not in self.eligible_clients:
                    continue
                client.tune_lambda()
                local_lam, local_acc = self.channel.receive("server", client.index, "lambda").payload
                local_best_accs[client.index] = local_acc
                local_best_lambdas[client.index] = local_lam

            lam = np.sum(local_best_lambdas)/np.sum(local_best_accs)
            self._compute_final_model(lam)

        if FlukeENV().get_eval_cfg().server:
            evals = self.evaluate(FlukeENV().get_evaluator(), self.test_set)
            self.notify("server_evaluation", round=self.rounds + 1, eval_type="global", evals=evals)

            for k,v in evals.items():
                self.notify("track_item", round=self.rounds+1, item=k, value=v)

       #self._notify("finalize")


class Fed2RC(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return ClientFed2RC

    def get_server_class(self) -> type[Server]:
        return ServerFed2RC
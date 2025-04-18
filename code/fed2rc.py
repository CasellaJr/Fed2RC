
from typing import Any, Generator, Iterable, Union

import numpy as np
import torch
from fluke import FlukeENV  # NOQA
from fluke.algorithms import CentralizedFL  # NOQA
from fluke.client import Client  # NOQA
from fluke.comm import Message  # NOQA
from fluke.data import FastDataLoader  # NOQA
from fluke.server import EarlyStopping, Server  # NOQA
from fluke.utils import OptimizerConfigurator  # NOQA
from rich.progress import track
from sklearn.linear_model import RidgeClassifierCV  # NOQA
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.rocket import Rocket

FEAT_CACHE = {}


def get_from_cache(client_id: int, seed: int) -> np.ndarray:
    if client_id in FEAT_CACHE and seed in FEAT_CACHE[client_id]:
        return FEAT_CACHE[client_id][seed]
    return None


def put_in_cache(client_id: int, seed: int, value: np.ndarray) -> None:
    if client_id not in FEAT_CACHE:
        FEAT_CACHE[client_id] = {}
    FEAT_CACHE[client_id][seed] = value


def transform(X: np.ndarray,
              client_id: int,
              seed: int,
              ppv_only: bool = True,
              caching: bool = True) -> np.ndarray:
    if caching:
        cached = get_from_cache(client_id, seed)
        if cached is not None:
            return cached

    rocket = Rocket(num_kernels=1, random_state=seed, n_jobs=1)
    rocket.fit(np.zeros((1, 1, X.shape[1])))

    if len(X.shape) <= 1:
        X = X.reshape(1, -1)

    X_transformed = rocket.transform(np.expand_dims(X, 1)).to_numpy()
    if ppv_only:
        X_transformed = X_transformed[:, 0:1]

    if caching:
        put_in_cache(client_id, seed, X_transformed)

    return X_transformed


def transform_seeds(X: np.ndarray,
                    client_id: int,
                    seeds: Iterable[int],
                    caching: bool = True,
                    use_torch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    Xnp = np.concatenate([transform(X, client_id, int(seed), caching=caching)
                         for seed in seeds], axis=1)
    return torch.tensor(Xnp, dtype=torch.float32) if use_torch else Xnp.astype(np.float32)


class RidgeTorchModel(torch.nn.Module):
    def __init__(self,
                 W: np.ndarray,
                 b: np.ndarray,
                 seeds: Iterable[int]):
        super().__init__()
        self.model = torch.nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
        self.seeds = seeds

        with torch.no_grad():
            self.model.weight.copy_(torch.tensor(W, dtype=torch.float32))
            if b is not None:
                self.model.bias.copy_(torch.tensor(b, dtype=torch.float32))

    def forward(self, X: torch.Tensor):
        X = transform_seeds(X, -1, self.seeds, use_torch=True, caching=False)
        output = self.model(X)
        return output


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
                 **kwargs: dict[str, Any]):
        super().__init__(index, train_set, test_set, None, None, 0, 0, 0, **kwargs)
        self.hyper_params.update(n_kernels=n_kernels, top_k=top_k)
        self.seeds: Iterable[int] = np.arange((self.index) * n_kernels,
                                              (self.index + 1) * n_kernels)
        self.converged: bool = False
        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def send_model(self):
        if self.converged:
            self.channel.send(Message((self.A, self.b), "Ab", self), self.server)
        else:
            self.channel.send(Message(self.top_k_idx, "seeds", self), self.server)

    def receive_model(self):
        self.converged = self.channel.receive(self, self.server, msg_type="converge").payload
        if not self.converged:
            msg = self.channel.receive(self, self.server, msg_type="seeds")
            self.seeds = msg.payload if self.server.rounds != 0 else self.seeds

    def fit(self, **kwargs: dict[str, Any]) -> float:
        X, y = self.train_set.tensors
        X, y = X.numpy(), y.numpy()
        X_trans = transform_seeds(X, self.index, self.seeds)

        if self.converged:
            y_onehot = np.eye(self.train_set.num_labels)[y]
            self.A = X_trans.T @ X_trans
            self.b = X_trans.T @ y_onehot
        else:
            ridge = RidgeClassifierCV(alphas=np.logspace(-3, 0, 100)).fit(X_trans, y)
            # Useful if client evaluation is needed
            self.model = RidgeTorchModel(ridge.coef_,
                                         ridge.intercept_,
                                         self.seeds)
            self.top_k_idx = self.seeds[np.argsort(-np.abs(ridge.coef_[0]))
                                        [:self.hyper_params.top_k]]

        return 0.0

    def tune_lambda(self):
        A_global, b_global = self.channel.receive(self, self.server, "Ab").payload
        A_minus_client = A_global - self.A
        b_minus_client = b_global - self.b

        X, y = self.train_set.tensors
        X, y = X.numpy(), y.numpy()
        X_trans = transform_seeds(X, self.index, self.seeds)

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
                          "lambda", self), self.server)


class ServerFed2RC(Server):

    def __init__(self,
                 model: RidgeTorchModel,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = False,
                 lr: float = 1.0,
                 tune_lambda: bool = False,
                 **kwargs: dict[str, Any]):
        super().__init__(None, test_set, clients, weighted, None, **kwargs)
        self.hyper_params.update(tune_lambda=tune_lambda)
        self.seeds = []
        self.converged = False

    def receive_client_models(self,
                              eligible: Iterable[Client],
                              **kwargs: dict[str, Any]) -> Generator[Any, None, None]:
        if not self.converged:
            for client in eligible:
                client_seeds = self.channel.receive(self, client, "seeds").payload
                yield client_seeds
        else:
            for client in eligible:
                client_Ab = self.channel.receive(self, client, "Ab").payload
                yield client_Ab

    def broadcast_model(self, eligible: Iterable[Client]) -> None:
        self.channel.broadcast(Message(self.converged, "converge", self), eligible)
        if not self.converged:
            self.channel.broadcast(Message(self.seeds, "seeds", self), eligible)

    def _compute_final_model(self, lam: float = 0.01) -> None:
        A_final = self.A + lam * np.eye(self.A.shape[0])
        W_final = np.linalg.solve(A_final, self.b)
        W_final = W_final / (np.linalg.norm(W_final, axis=0, keepdims=True) + 1e-10)
        self.model = RidgeTorchModel(W_final.T, None, self.seeds)

    def aggregate(self,
                  eligible: Iterable[Client],
                  client_seeds_ab: Generator[Any, None, None]) -> None:
        if not self.converged:
            seeds = np.concatenate(list(client_seeds_ab), axis=0)
            seeds = np.unique(seeds)
            self.converged = (len(self.seeds) == len(seeds)) and np.all(self.seeds == seeds)
            self.seeds = seeds
            self._notify_track_item(item="seeds", value=len(self.seeds))
        else:
            A_global = np.zeros((len(self.seeds), len(self.seeds)))
            b_global = None

            for client_Ab in client_seeds_ab:
                A, b = client_Ab
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
            self.channel.broadcast(Message((self.A, self.b), "Ab", self), self.clients)
            local_best_lambdas = np.zeros(len(self.clients))
            local_best_accs = np.zeros(len(self.clients))
            for client in track(self.clients, description="Validating lambda...", transient=True):
                client.tune_lambda()
                local_lam, local_acc = self.channel.receive(self, client, "lambda").payload
                local_best_accs[client.index] = local_acc
                local_best_lambdas[client.index] = local_lam

            lam = np.sum(local_best_lambdas)/np.sum(local_best_accs)
            self._compute_final_model(lam)

        if FlukeENV().get_eval_cfg().server:
            evals = self.evaluate(FlukeENV().get_evaluator(), self.test_set)
            self._notify_evaluation(self.rounds + 1, "global", evals)

        self._notify_finalize()


class Fed2RC(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return ClientFed2RC

    def get_server_class(self) -> type[Server]:
        return ServerFed2RC

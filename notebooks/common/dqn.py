"""DQN con PyTorch (D-0.2): interfaz por implementar.

Se implementa aquí y no en el notebook para poder lanzar barridos con `run_grid`.
Los valores por defecto son los del notebook de clase `2-cart_pole_DQN`, solo como punto de partida.
"""

from dataclasses import dataclass

from .agents import N_ACTIONS, OBS_DIM
from .protocol import N_STEPS


@dataclass
class DQNConfig:
    gamma: float = 0.9
    lr: float = 1e-3
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 20_000
    batch_size: int = 128
    hidden: int = 100
    n_episodes: int = 500


class DQNAgent:
    def __init__(self, cfg, obs_dim=OBS_DIM, n_actions=N_ACTIONS, seed=None):
        raise NotImplementedError

    def act(self, obs, greedy=False):
        """Acción epsilon-greedy; con `greedy=True`, la de mayor Q."""
        raise NotImplementedError

    def remember(self, obs, action, reward, next_obs, done):
        """Guarda la transición en la memoria de experiencias."""
        raise NotImplementedError

    def learn(self):
        """Un paso de entrenamiento con un minibatch de la memoria."""
        raise NotImplementedError

    def end_episode(self):
        """Decaimiento de epsilon."""
        raise NotImplementedError

    def save(self, name):
        """Guarda los pesos en `results/models/<name>.pt`."""
        raise NotImplementedError


def train_dqn(cfg, market, seed=0, n_steps=N_STEPS):
    """Entrena con una serie nueva por episodio, usando `train_series_seed(seed, episodio)`.

    Devuelve (agente, recompensa por episodio, tiempo en segundos).
    """
    raise NotImplementedError

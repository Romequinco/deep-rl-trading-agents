"""A3C con PyTorch (D-0.2): interfaz por implementar.

`worker` tiene que estar en este módulo: en Windows los procesos se crean con `spawn` y no pueden
importar funciones definidas en una celda de un notebook. Cada proceso debe llamar a
`torch.set_num_threads(1)` y a `set_seed` con una semilla propia. Los valores por defecto de `gamma`,
`lr` y `hidden` son los del notebook de clase `3-cart_pole_actor_critic`; `n_workers` deja un núcleo
libre en un equipo de 4, y `t_max` y `entropy_coef` son solo un punto de partida.
"""

from dataclasses import dataclass

from .agents import N_ACTIONS, OBS_DIM
from .protocol import N_STEPS


@dataclass
class A3CConfig:
    gamma: float = 0.99
    lr: float = 0.01
    hidden: int = 128
    n_workers: int = 3
    t_max: int = 20
    entropy_coef: float = 0.01
    n_episodes: int = 500


class A3CAgent:
    def __init__(self, cfg, obs_dim=OBS_DIM, n_actions=N_ACTIONS):
        raise NotImplementedError

    def act(self, obs, greedy=False):
        """Muestrea de la política; con `greedy=True`, la acción más probable."""
        raise NotImplementedError

    def save(self, name):
        """Guarda los pesos en `results/models/<name>.pt`."""
        raise NotImplementedError


def worker(rank, shared_model, optimizer, cfg, market, seed, n_steps, log):
    """Bucle de un proceso: copia los pesos compartidos, juega `t_max` pasos y aplica sus gradientes."""
    raise NotImplementedError


def train_a3c(cfg, market, seed=0, n_steps=N_STEPS):
    """Lanza `cfg.n_workers` procesos, cada uno con sus propias series de `train_series_seed`.

    Devuelve (agente, recompensa por episodio, tiempo en segundos).
    """
    raise NotImplementedError

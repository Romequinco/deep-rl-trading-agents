"""Baselines e interfaz común de los agentes.

Todo agente expone `act(obs, greedy=False) -> int`, donde `obs` es la observación continua del entorno.
Con eso basta para `evaluate()`, así que baselines, DQN y A3C se miden con el mismo código.
"""

import numpy as np

from .protocol import evaluate, summarize

SELL, HOLD, BUY = 0, 1, 2
N_ACTIONS = 3
OBS_DIM = 2  # (tendencia, posición)


class RandomAgent:
    """Línea base del taller: acción al azar en cada paso."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, greedy=False):
        return int(self.rng.integers(N_ACTIONS))


class BuyHoldAgent:
    """Compra en el primer paso y mantiene hasta el final (comprar con posición abierta no hace nada)."""

    def act(self, obs, greedy=False):
        return BUY


class NoTradeAgent:
    """No opera nunca. En el Taller 1 batió a buy & hold en los mercados 2-4, así que es una referencia obligada."""

    def act(self, obs, greedy=False):
        return HOLD


class TrendRuleAgent:
    """Regla fija: entra si la tendencia supera `threshold` y sale si cae por debajo de `-threshold`."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def act(self, obs, greedy=False):
        if obs[0] > self.threshold:
            return BUY
        return SELL if obs[0] < -self.threshold else HOLD


class ReversionRuleAgent(TrendRuleAgent):
    """Regla fija opuesta: entra tras una caída mayor que `threshold` y sale tras una subida igual."""

    def act(self, obs, greedy=False):
        if obs[0] < -self.threshold:
            return BUY
        return SELL if obs[0] > self.threshold else HOLD


def policy_string(agent, position, n=11, k_max=2.5):
    """Acción greedy sobre una rejilla de tendencias entre ±`k_max` σ: 'V', 'M' o 'C' por punto. Las entradas
    añadidas al estado, si las hay, se fijan en su valor medio."""
    letters = {SELL: "V", HOLD: "M", BUY: "C"}
    base = np.array(agent.neutral_obs(position), dtype=np.float32)
    out = []
    for t in np.linspace(-k_max, k_max, n) * agent.scales[0]:
        obs = base.copy()
        obs[0] = t
        out.append(letters[agent.act(obs, greedy=True)])
    return "".join(out)


RULES = {"regla_tendencia": TrendRuleAgent, "regla_reversion": ReversionRuleAgent}


def baseline_agents():
    """Baselines con los nombres de `RESULT_COLUMNS`. La semilla es la del Taller 1, que da los mismos números."""
    return {"aleatorio": RandomAgent(seed=0), "buy_hold": BuyHoldAgent(), "no_operar": NoTradeAgent()}


def evaluate_reference(cfg):
    """Evalúa un baseline o una regla para `run_grid`. La `curva` es el retorno de cada serie."""
    name = cfg["agente"]
    agent = RULES[name](cfg["umbral"]) if name in RULES else baseline_agents()[name]
    df = evaluate(agent, cfg["mercado"], split=cfg["split"])
    return {"fila": summarize(name, cfg["mercado"], df), "curva": df["retorno"].to_numpy()}

import typing as t

import numpy as np

from .operations import I, X, Y, Z, tensorproduct
from .types import NBitOperation, NBitState


def add_noise[N](state: NBitState[N], /) -> NBitState[N]:
    """
    Random unitary rotation on 1 bit.
    """
    n = len(state).bit_length() - 1
    gates = [I for _ in range(n)]
    i = np.random.choice(n)
    gates[i] = random_unitary()
    noise_op = tensorproduct(*gates)
    return noise_op @ state


def random_unitary() -> NBitOperation[t.Literal[1]]:
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    return np.sin(theta) * (np.cos(phi) * X + np.sin(phi) * Y) + np.cos(theta) * Z

import functools
import typing as t

import numpy as np

from .types import NBitOperation, NBitState


I: NBitOperation[t.Literal[1]] = np.eye(2, dtype=np.complex128)
X: NBitOperation[t.Literal[1]] = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y: NBitOperation[t.Literal[1]] = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z: NBitOperation[t.Literal[1]] = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def onehot(*bits: t.Literal[0, 1]) -> NBitState:
    out = np.zeros((2, ) * len(bits), dtype=np.complex128)
    out[bits] = 1
    return out.ravel()


@t.overload
def tensorproduct(*states: NBitState) -> NBitState: ...
@t.overload
def tensorproduct(*states: NBitOperation) -> NBitOperation: ...

def tensorproduct(*states):
    return functools.reduce(np.kron, states)


def fidelity[N](a: NBitState[N], b: NBitState[N], /) -> float:
    return np.abs(np.conjugate(a) @ b) ** 2


def phase_estimation[N](
    measurement: NBitOperation[N], state: NBitState[N],
) -> tuple[t.Literal[1, -1], NBitState[N]]:
    """
    Parameters
    ----------
    measurment
        Eigenvalues should be +-1

    Returns
    -------
    Measurement result & post measurement state.
    """    
    I_n: NBitOperation[N] = np.eye(len(state), dtype=state.dtype)
    projection_plus1 = (I_n + measurement) / 2
    state_plus1 = projection_plus1 @ state  # unnormalize
    norm = np.linalg.norm(state_plus1)
    p_plus1 = norm ** 2
    if np.random.uniform() < p_plus1:
        return 1, state_plus1 / norm
    return -1, normalize(state - state_plus1)


def normalize[N](state: NBitState[N]) -> NBitState[N]:
    return state / np.linalg.norm(state)

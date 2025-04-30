from __future__ import annotations

import functools
import typing as t

import numpy as np
import numpy.typing as npt
import typer


def main():
    # Steane Code
    # Shor Code
    # 5 qubit Code
    # codecs = RepetitionCode()
    codecs = ShorCode()

    original = random_unitary() @ onehot(0)

    codeword = codecs.encode(original)
    codeword = add_noise(codeword)
    corrected = codecs.correct(codeword)
    result = codecs.decode(corrected)
    print(fidelity(result, original))


class QuantumCodecs[N: int, K: int](t.Protocol):
    def encode(self, raw: NBitState[K], /) -> NBitState[N]: ...
    def correct(self, code: NBitState[N], /) -> NBitState[N]: ...
    def decode(self, code: NBitState[N], /) -> NBitState[K]: ...


class StabilizerCode[N: int, K: int](QuantumCodecs[N, K]):

    @t.override
    def encode(self, raw, /):
        arr = raw @ self.eigenstates  # (2**K,) @ (2**K, 2**N) -> (2**N,)
        return t.cast('NBitState[N]', arr)

    @t.override
    def correct(self, code, /):
        # 1. Detect syndrome
        syndromes: list[t.Literal[1, -1]] = []
        for g in self.stabilizer_generators:            
            s, code = phase_estimation(g, code)
            syndromes.append(s)

        # 2. collapse code to syndrome eigenstate
        # 3. Map syndrome to errors
        # 4. Correct error with least weight (more likely to happen in most noise model)
        correction = self.get_correction(syndromes)
        return correction @ code

    @t.override
    def decode(self, code, /):
        x = self.eigenstates @ np.conjugate(code)  # (2**K, 2**N) @ (2**N, ) -> (2**K,)
        x = np.conjugate(x)
        return t.cast('NBitState[K]', x)

    @functools.cached_property
    def eigenstates(self) -> t.Sequence[NBitState[N]]:  # len 2**K
        ...

    @functools.cached_property
    def stabilizer_generators(self) -> t.Sequence[NBitOperation[N]]:  # len N - K
        ...

    def get_correction(self, syndrome: t.Sequence[t.Literal[1, -1]], /) -> NBitOperation[N]:  # syndrome len N - K
        ...


class RepetitionCode(StabilizerCode[t.Literal[3], t.Literal[1]]):

    @functools.cached_property
    def eigenstates(self):  # shape (2, 8)
        return [
            onehot(0, 0, 0),
            onehot(1, 1, 1),
        ]

    @functools.cached_property
    def stabilizer_generators(self):
        return [
            tensorproduct(Z, Z, I),
            tensorproduct(I, Z, Z),
        ]

    def get_correction(self, syndrome):
        match tuple(syndrome):
            case ( 1,  1): return tensorproduct(I, I, I)
            case (-1,  1): return tensorproduct(X, I, I)
            case (-1, -1): return tensorproduct(I, X, I)
            case ( 1, -1): return tensorproduct(I, I, X)

        t.assert_never(syndrome)


class ShorCode(StabilizerCode[t.Literal[9], t.Literal[1]]):

    @functools.cached_property
    def eigenstates(self):  # shape (2, 512)
        s0 = tensorproduct(
            (onehot(0, 0, 0) + onehot(1, 1, 1)) / (2 ** 0.5),
            (onehot(0, 0, 0) + onehot(1, 1, 1)) / (2 ** 0.5),
            (onehot(0, 0, 0) + onehot(1, 1, 1)) / (2 ** 0.5),
        )
        s1 = tensorproduct(
            (onehot(0, 0, 0) - onehot(1, 1, 1)) / (2 ** 0.5),
            (onehot(0, 0, 0) - onehot(1, 1, 1)) / (2 ** 0.5),
            (onehot(0, 0, 0) - onehot(1, 1, 1)) / (2 ** 0.5),
        )
        return [s0, s1]

    @functools.cached_property
    def stabilizer_generators(self):
        return [
            tensorproduct(Z, Z, I, I, I, I, I, I, I),
            tensorproduct(I, Z, Z, I, I, I, I, I, I),
            tensorproduct(I, I, I, Z, Z, I, I, I, I),
            tensorproduct(I, I, I, I, Z, Z, I, I, I),
            tensorproduct(I, I, I, I, I, I, Z, Z, I),
            tensorproduct(I, I, I, I, I, I, I, Z, Z),
            tensorproduct(X, X, X, X, X, X, I, I, I),
            tensorproduct(I, I, I, X, X, X, X, X, X),
        ]

    def get_correction(self, syndrome):
        r = RepetitionCode()
        xs = tensorproduct(
            r.get_correction(syndrome[:2]),
            r.get_correction(syndrome[2:4]),
            r.get_correction(syndrome[4:6]),
        )
        match tuple(syndrome[6:]):
            case ( 1,  1): zs = tensorproduct(I, I, I, I, I, I, I, I, I)
            case (-1,  1): zs = tensorproduct(Z, I, I, I, I, I, I, I, I)
            case (-1, -1): zs = tensorproduct(I, I, I, Z, I, I, I, I, I)
            case ( 1, -1): zs = tensorproduct(I, I, I, I, I, I, Z, I, I)
            case _: t.assert_never(syndrome)

        return xs @ zs


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


def phase_estimation[N](
    measurement: NBitOperation[N], state: NBitState[N], /,
) -> tuple[t.Literal[1, -1], NBitState[N]]:
    """
    Perform measurement whose eigenvalues are +-1
    return result & post measurement state.
    """
    In: NBitOperation[N] = np.eye(len(state), dtype=state.dtype)
    projection_plus1: NBitOperation[N] = (In + measurement) / 2
    post = projection_plus1 @ state
    p_plus1 = np.linalg.norm(post) ** 2
    if np.random.uniform() < p_plus1:
        return 1, normalize(post)

    projection_minus1 = (In - measurement) / 2
    return -1, normalize(projection_minus1 @ state)


def normalize[N](state: NBitState[N]) -> NBitState[N]:
    return state / np.linalg.norm(state)


def random_unitary() -> NBitOperation[t.Literal[1]]:
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    return np.sin(theta) * (np.cos(phi) * X + np.sin(phi) * Y) + np.cos(theta) * Z


I: NBitOperation[t.Literal[1]] = np.eye(2, dtype=np.complex128)
X: NBitOperation[t.Literal[1]] = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y: NBitOperation[t.Literal[1]] = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z: NBitOperation[t.Literal[1]] = np.array([[1, 0], [0, -1]], dtype=np.complex128)


Scalar = int | float | complex | np.integer | np.floating | np.complexfloating


class NBitState[N](npt.NDArray[np.complexfloating]):
    def __add__(self, other: NBitState[N]) -> t.Self: ...
    def __sub__(self, other: NBitState[N]) -> t.Self: ...
    def __mul__(self, other: Scalar) -> t.Self: ...
    def __truediv__(self, scalar: Scalar) -> t.Self: ...


class NBitOperation[N](npt.NDArray[np.complexfloating]):
    def __add__(self, other: NBitOperation[N]) -> t.Self: ...
    def __sub__(self, other: NBitOperation[N]) -> t.Self: ...
    def __truediv__(self, scalar: Scalar) -> t.Self: ...
    @t.overload
    def __matmul__(self, other: NBitOperation[N]) -> t.Self: ...
    @t.overload
    def __matmul__(self, other: NBitState[N]) -> NBitState[N]: ...


if __name__ == "__main__":
    typer.run(main)

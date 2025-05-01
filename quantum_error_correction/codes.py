import functools
import typing as t

import numpy as np

from .operations import I, X, Z, onehot, phase_estimation, tensorproduct
from .types import NBitOperation, NBitState


class QuantumCodecs[N: int, K: int](t.Protocol):
    def encode(self, raw: NBitState[K], /) -> NBitState[N]: ...
    def correct(self, code: NBitState[N], /) -> NBitState[N]: ...
    def decode(self, code: NBitState[N], /) -> NBitState[K]: ...


class IdentityCode(QuantumCodecs[t.Literal[1], t.Literal[1]]):
    def encode(self, raw):
        return raw
    def correct(self, code, /):
        return code
    def decode(self, code, /):
        return code

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

    @t.override
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

import typer

from quantum_error_correction.codes import ShorCode
from quantum_error_correction.noises import add_noise, random_unitary
from quantum_error_correction.operations import fidelity, onehot


def main():
    # TODO select code & noise
    codecs = ShorCode()

    original = random_unitary() @ onehot(0)

    codeword = codecs.encode(original)
    codeword = add_noise(codeword)
    corrected = codecs.correct(codeword)
    result = codecs.decode(corrected)
    print(fidelity(result, original))


if __name__ == "__main__":
    typer.run(main)

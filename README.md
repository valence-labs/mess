# MESS: Modern Electronic Structure Simulations

> [!IMPORTANT]
> :hammer: :skull: :warning: :wrench:\
> This project is a constantly evolving work in progress.\
> Expect bugs and surprising performance cliffs.\
> :hammer: :skull: :warning: :wrench:

[![docs](https://img.shields.io/badge/MESS-docs-blue?logo=bookstack)](https://valence-labs.github.io/mess)
[![arXiv](https://img.shields.io/badge/arXiv-2406.03121-b31b1b.svg)](https://arxiv.org/abs/2406.03121)
[![unit tests](https://github.com/valence-labs/mess/actions/workflows/unittest.yaml/badge.svg)](https://github.com/valence-labs/mess/actions/workflows/unittest.yaml)
[![pre-commit checks](https://github.com/valence-labs/mess/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/valence-labs/mess/actions/workflows/pre-commit.yaml)

## Motivation

MESS is primarily motivated by the need to demystify the underpinnings of electronic
structure simulations. The target audience is the collective molecular machine learning
community. We identify this community as anyone working towards accelerating our
understanding of atomistic processes by combining physical models (e.g. density
functional theory) with methods that learn from data (e.g. deep neural networks).

## Minimal Example

Calculate the ground state energy of a single water molecule using the 6-31g basis set
and the [local density approximation (LDA)](https://en.wikipedia.org/wiki/Local-density_approximation):
```python
from mess import Hamiltonian, basisset, minimise, molecule

mol = molecule("water")
basis = basisset(mol, basis_name="6-31g")
H = Hamiltonian(basis, xc_method="lda")
E, C, sol = minimise(H)
E
```

## License

The reader is encouraged to fork, edit, remix, and use the contents however they find
most useful. All content is covered by the permissve [MIT License](./LICENSE) to
encourage this. Our aim is to encourage a truly interdisciplinary approach to accelerate
our understanding of molecular scale processes.

## Installing

We recommend installing directly from the main branch from github and sharing any
feedback as [issues](https://github.com/valence-labs/mess/issues).

```
pip install git+https://github.com/valence-labs/mess.git
```

Requires Python 3.11+ and we recommend [installing JAX](https://jax.readthedocs.io/en/latest/installation.html) for your target system (e.g. CPU, GPU, etc).


## Citation
If you found this library to be useful in academic work, then please cite our
[arXiv paper](https://arxiv.org/abs/2406.03121)
```
@misc{helal2024mess,
      title={MESS: Modern Electronic Structure Simulations},
      author={Hatem Helal and Andrew Fitzgibbon},
      year={2024},
      eprint={2406.03121},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

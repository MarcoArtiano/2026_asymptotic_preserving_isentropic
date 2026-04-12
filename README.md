# Asymptotic-Preserving and Well-Balanced Linearly Implicit IMEX Schemes for the Anelastic Limit of the Isentropic Euler Equations with Gravity

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/TODO)](https://zenodo.org/doi/TODO)

This repository contains information and code to reproduce the results presented in the article

```bibtex
@online{artiano2026asymptotic,
  title={Asymptotic-Preserving and Well-Balanced Linearly Implicit IMEX Schemes
         for the Anelastic Limit of the Isentropic Euler Equations with Gravity},
  author={Artiano, Marco and Samantaray, Saurav
            and Ranocha, Hendrik},
  year={2026},
  month={TODO},
  eprint={TODO},
  eprinttype={arxiv},
  eprintclass={TODO}
}
```

If you find these results useful, please cite the article mentioned above. If you use the implementations provided here, please **also** cite this repository as

```bibtex
@misc{artiano2026asymptoticeRepo,
  title={Reproducibility repository for
         "{A}symptotic-Preserving and Well-Balanced Linearly Implicit IMEX Schemes
           for the Anelastic Limit of the Isentropic Euler Equations with Gravity"},
  author={Artiano, Marco and Samantaray, Saurav
         and Ranocha, Hendrik},
  year={2026},
  howpublished={\url{https://github.com/MarcoArtiano/2026_asymptotic_preserving_isentropic}},
  doi={TODO}
}
```

## Abstract
We consider the compressible Euler system with anelastic scaling modeling isentropic flows under the influence of gravity. 
In the zero-Mach-number limit, the solution of the compressible Euler system converges to a variable density anelastic incompressible limit system. 
In this work, we present the design and analysis of a class of higher-order linearly implicit IMEX Runge-Kutta schemes that are asymptotic preserving, i.e., they respect the transitory nature of the governing equations in the limit. 
The presence of gravitational potential warrants the incorporation of the well-balancing property. 
The scheme is developed as a novel combination of a penalization of a linear steady state, a finite-volume balance-preserving reconstruction, and a source term discretization preserving steady states. 
The penalization plays a crucial role in obtaining a linearly implicit scheme, and well-balanced flux-source discretization ensures accuracy in very low Mach number regimes. 
Some results of numerical case studies are presented to corroborate the theoretical assertions.


## Numerical experiments
To reproduce the numerical experiments presented in this article, you need to install Julia. The numerical experiments presented in this article were performed using Julia v1.10.6.

First, you need to download this repository, e.g., by cloning it with git or by downloading an archive via the GitHub interface. Then, you need to start Julia in the code directory of this repository and follow the instructions described in the README.md file therein.

## Authors
- Marco Artiano
- [Saurav Samantaray](https://sauravsray.github.io/)
- [Hendrik Ranocha](https://ranocha.de/) (Johannes Gutenberg University Mainz, Germany)

## License

The code in this repository is published under the MIT license, see the `LICENSE` file.

## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!

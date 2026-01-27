# Shamrock features

Here is a somewhat Exhaustive list of shamrock's features, do not hesitate to raise an issue if one appear to be missing.
This page was made in order to list the features of the code as well as properly attributing contribution and avoid having multiple peoples working on the same features.

We list the features by categories as well as their status which can be any of:
![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)
![Ok](https://img.shields.io/badge/Ok-yellowgreen)
![WIP](https://img.shields.io/badge/WIP-yellow) (Work in progress)
![Broken](https://img.shields.io/badge/Broken-red)
This page also trace the contributor who made the contribution as well as the corresponding paper to cite for each features.
If any feature is notated with
![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical)
please wait for the corresponding feature to be published before publishing anything using it.

## Physical

### SPH model

#### Core features

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Gas solver | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| [Sink particles](./features/sph/sinks.md) | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) | |
| Pseudo-Newtonian <br> corrections | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) | [![PR - #319](https://img.shields.io/badge/PR-%23319-brightgreen?logo=github)](https://github.com/Shamrock-code/Shamrock/pull/319) |
| MHD solver | ![WIP](https://img.shields.io/badge/WIP-yellow)  |  [Yona Lapeyre](https://github.com/y-lapeyre) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) | [![PR - #707](https://img.shields.io/badge/PR-%23707-brightgreen?logo=github)](https://github.com/Shamrock-code/Shamrock/pull/707) |


| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| [On the fly-plots](./usermanual/plotting.md) | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) | [![PR - #623](https://img.shields.io/badge/PR-%23623-brightgreen?logo=github)](https://github.com/Shamrock-code/Shamrock/pull/623) |
| [Conformance with Phantom](./features/sph/conformance_phantom.md) | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) <br> & [Yona Lapeyre](https://github.com/y-lapeyre) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| [Setup graph](./features/sph/setup_graph.md) | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) | [![PR - #593](https://img.shields.io/badge/PR-%23593-brightgreen?logo=github)](https://github.com/Shamrock-code/Shamrock/pull/593) |
| [Shearing box](./features/sph/shearing_box.md) | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) | |
| Periodic box | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) | |

#### Shock handling mechanisms

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Constant $\alpha_{AV}$ | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) <br> & [Yona Lapeyre](https://github.com/y-lapeyre) | | |
| MM97 $\alpha_{AV}$ | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl)| [![Nasa ads - MM97](https://img.shields.io/badge/Nasa_ads-MM97-blue)](https://ui.adsabs.harvard.edu/abs/1997JCoPh.136...41M/abstract) | |
| CD10 $\alpha_{AV}$ | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl)| [![Nasa ads - CD10](https://img.shields.io/badge/Nasa_ads-CD10-blue)](https://ui.adsabs.harvard.edu/abs/2010MNRAS.408..669C/abstract) | |
| $\alpha$-disc viscosity | ![Ok](https://img.shields.io/badge/Ok-yellowgreen) |  [Yona Lapeyre](https://github.com/y-lapeyre)|  | Requires the warp diffusion test to fully validate |

#### Equations of state

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Isothermal | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Yona Lapeyre](https://github.com/y-lapeyre) | | |
| Adiabatic | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | | |
| Isothermal - LP07 | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Yona Lapeyre](https://github.com/y-lapeyre) | | [![PR - #361](https://img.shields.io/badge/PR-%23361-brightgreen?logo=github)](https://github.com/Shamrock-code/Shamrock/pull/361)  |
| Isothermal - FA14 | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | | |
| Ideal Fermi gas | ![Ok](https://img.shields.io/badge/Ok-yellowgreen)  |  [David Fang](https://github.com/DavidFang03) | | [![PR - #1405](https://img.shields.io/badge/PR-%231405-brightgreen?logo=github)](https://github.com/Shamrock-code/Shamrock/pull/1405) |
| Tillotson | ![Ok](https://img.shields.io/badge/Ok-yellowgreen)  |  [David Fang](https://github.com/DavidFang03) | | [![PR - #1445](https://img.shields.io/badge/PR-%231445-brightgreen?logo=github)](https://github.com/Shamrock-code/Shamrock/pull/1445) |

### Godunov model

#### Principal components

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Ramses solver | ![Ok](https://img.shields.io/badge/Ok-yellowgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) & [Thomas Guillet](https://github.com/thomasguillet) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) | Needs some polishing to be considered production ready |
| Refinement handling | ![Ok](https://img.shields.io/badge/Ok-yellowgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) |  |
| Multifluid dust | ![WIP](https://img.shields.io/badge/WIP-yellow)  |  [Léodasce Sewanou](https://github.com/Akos299) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) | [![PR - #636](https://img.shields.io/badge/PR-%23636-yellow?logo=github)](https://github.com/Shamrock-code/Shamrock/pull/636) |

#### Refinement criterions

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Mass based refinement | ![Ok](https://img.shields.io/badge/Ok-yellowgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) |  |
| Pseudo-gradient refinement | ![WIP](https://img.shields.io/badge/WIP-yellow)  |  [Léodasce Sewanou](https://github.com/Akos299) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) |  |
| Modified second derivative refinement | ![WIP](https://img.shields.io/badge/WIP-yellow)  |  [Léodasce Sewanou](https://github.com/Akos299) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) |  |

#### Slope limiters

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| None | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) |  |
| Minmod | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) |  |
| VanLeer | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) |  |
| Symmetrized VanLeer | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) & [Thomas Guillet](https://github.com/thomasguillet) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) |  |

#### Riemann solvers

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Rusanov | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) & [Thomas Guillet](https://github.com/thomasguillet) |  |  |
| HLL | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) & [Thomas Guillet](https://github.com/thomasguillet) |  |  |
| HLLC | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Léodasce Sewanou](https://github.com/Akos299) |  |
| Dusty HLL | ![WIP](https://img.shields.io/badge/WIP-yellow) |  [Léodasce Sewanou](https://github.com/Akos299) |  |
| HB dust solver | ![WIP](https://img.shields.io/badge/WIP-yellow) |  [Léodasce Sewanou](https://github.com/Akos299) |  |

### Zeus model

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Core solver | ![Ok](https://img.shields.io/badge/Ok-yellowgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical) | Needs some polishing to be considered production ready |

### NBody FMM solver

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Core solver | ![WIP](https://img.shields.io/badge/WIP-yellow) |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Wait for the paper !](https://img.shields.io/badge/Wait_for_the_paper_!-critical)  | WIP of a N-Body FMM self-gravity solver, physically correct but not usable for any production runs yet. |

## Framework

### Software

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Python integration | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Test library | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| CI/CD | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |

### Shamrock internal libraries

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Shamalgs | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shambackends | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shambase | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shambindings | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shamcmdopt | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shamcomm | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shammath | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shammodels | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shamphys | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shamrock | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shamsys | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shamtest | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shamtree | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |
| Shamunits | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713) | |

### Components

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Patch system | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen) |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713)  |
| Sparse communications | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen) |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713)  |
| Radix Tree | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen) |  [Timothée David--Cléris](https://github.com/tdavidcl) | [![Nasa ads](https://img.shields.io/badge/nasa_ADS-Shamrock1-blue.svg)](https://ui.adsabs.harvard.edu/abs/arXiv:2503.09713)  |

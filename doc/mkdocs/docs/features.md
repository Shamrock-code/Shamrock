# Shamrock features

Here is a somewhat Exhaustive list of shamrock's features, do not hesitate to raise an issue if one appear to be missing.
This page was made in order to list the features of the code as well as properly attributing contribution and avoid having multiple peoples working on the same features.

We list the features by categories as wel as their status which can be any of:
![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)
![Ok](https://img.shields.io/badge/Ok-yellowgreen)
![WIP](https://img.shields.io/badge/WIP-yellow) (Work in progress)
![Broken](https://img.shields.io/badge/Broken-red)
This page also trace the contributor who made the contributiona as well as the corresponding paper to cite for each features.
If any feature is notated with
![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical)
please wait for the corresponding feature to be published before publishing anything using it.

## Physical

### SPH model

#### Core features

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Gas solver | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Sink particles | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Pseudo-Newtonian <br> corrections | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | [![PR - #319](https://img.shields.io/badge/PR-%23319-brightgreen?logo=github)](https://github.com/tdavidcl/Shamrock/pull/319) |
| MHD solver | ![WIP](https://img.shields.io/badge/WIP-yellow)  |  [Yona Lapeyre](https://github.com/y-lapeyre) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | [![PR - #707](https://img.shields.io/badge/PR-%23707-yellow?logo=github)](https://github.com/tdavidcl/Shamrock/pull/707) |


| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| [On the fly-plots](./usermanual/plotting.md) | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | [![PR - #623](https://img.shields.io/badge/PR-%23623-brightgreen?logo=github)](https://github.com/tdavidcl/Shamrock/pull/623) |
| Conformance with Phantom | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) <br> & [Yona Lapeyre](https://github.com/y-lapeyre) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Setup graph | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shearing box | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Periodic box | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |

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
| Isothermal - LP07 | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Yona Lapeyre](https://github.com/y-lapeyre) | | [![PR - #361](https://img.shields.io/badge/PR-%23361-brightgreen?logo=github)](https://github.com/tdavidcl/Shamrock/pull/361)  |
| Isothermal - FA14 | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | | |

### Godunov model

#### Principal components

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Ramses solver | ![Ok](https://img.shields.io/badge/Ok-yellowgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | Needs some polishing to be considered production ready |
| Refinement handling | ![Ok](https://img.shields.io/badge/Ok-yellowgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) |  |
| Multifluid dust | ![WIP](https://img.shields.io/badge/WIP-yellow)  |  [Léodasce Sewanou](https://github.com/Akos299) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | [![PR - #636](https://img.shields.io/badge/PR-%23636-yellow?logo=github)](https://github.com/tdavidcl/Shamrock/pull/636) |

#### Refinement criterions

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Mass based refinement | ![Ok](https://img.shields.io/badge/Ok-yellowgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) |  |
| Pseudo-gradient refinement | ![WIP](https://img.shields.io/badge/WIP-yellow)  |  [Léodasce Sewanou](https://github.com/Akos299) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) |  |

#### Slope limiters

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| None | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) |  |
| Minmod | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) |  |
| VanLeer | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) |  |
| Symmetrized VanLeer | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) |  |

#### Riemann solvers

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Rusanov | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) |  |  |
| HLL | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) |  |  |
| HLLC | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Léodasce Sewanou](https://github.com/Akos299) |  |

### Zeus model

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Core solver | ![Ok](https://img.shields.io/badge/Ok-yellowgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | Needs some polishing to be considered production ready |

### NBody FMM solver

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Core solver | ![WIP](https://img.shields.io/badge/WIP-yellow) |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical)  | WIP of a N-Body FMM self-gravity solver, physically correct but not usable for any production runs yet. |

## Framework

### Software

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Python integration | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Test library | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| CI/CD | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | Needs to be extended when the code will be public |

### Shamrock internal libraries

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Shamalgs | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shambackends | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shambase | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shambindings | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shamcmdopt | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shamcomm | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shammath | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shammodels | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shamphys | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shamrock | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shamsys | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shamtest | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shamtree | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |
| Shamunits | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen)  |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical) | |

### Components

| Feature  | Status   | Contributor / Maintainer  |  Paper to cite | Details |
| --- | --- | --- | --- |  --- |
| Patch system | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen) |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical)  |
| Sparse communications | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen) |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical)  |
| Radix Tree | ![Production ready](https://img.shields.io/badge/Production_ready-brightgreen) |  [Timothée David--Cléris](https://github.com/tdavidcl) | ![Do not cite yet !](https://img.shields.io/badge/Do_not_cite_yet_!-critical)  |

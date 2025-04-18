# MDR-FEP
----

This is the repository accompanying my Master of Arts thesis in Chemistry. The focus of my thesis was using molecular dynamics (MD) and the Rosetta software to predict the effects of mutations to computationally designed protein binders (Cao et al. March, 2022. https://doi.org/10.1038/s41586-022-04654-9). We used a combination of MD, Rosetta, and free energy perturbation to quantify the binding effects of each mutation. This method, called MDR-FEP (Molecular Dynamics Rosetta Free Energy Perturbation) is first described by Wells. et al (February, 2023. https://doi.org/10.1002/prot.26477).

## The standard pipeline for MDR-FEP is to first carry out equilibrium MD of the WT sequence.
This involves:
  I. Energy minimization
  II. Equilibration
  III. Production MD
  IV. PBC correction
  V. Extracting frames
  For BOTH the binder and the binder-target structures independently.

With this completed you have an ensemble of WT structures that you can feed to Rosetta for fixed-backbone sidechain repacking and scoring.

## Before carrying out the Rosetta repacking and scoring you need an idea of what mutations you need an idea of what mutations you want to carry out, and what distance you want to repack each residue within. 5 Å is a conservative selection, as it repacks a little bit of the structure while being carried out quickly.



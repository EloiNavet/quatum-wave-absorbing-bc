# Modeling Absorbing Boundary Conditions for the Schrödinger Equation

This project is part of the [`Modeling Seminar`](https://msiam.imag.fr/m2siam_ue/gbx9am19/) course by [Brigitte Bidégaray-Fesquet](https://edp-ljk.imag.fr/author/brigitte-bidegaray-fesquet/) and [Clément Jourdana](https://membres-ljk.imag.fr/Clement.Jourdana/) at the Université Grenoble Alpes. The work was carried out by **Linnea Hallin**, **Maxime Renard**, **Nicolas Roblet** and **Éloi Navet** (me), during the 2023-2024 Fall semester. It focuses on the implementation and comparison of different approaches for modeling *Absorbing Boundary Conditions* (ABC) and *Perfectly Matched Layers* (PML) for solving quantum wave equations like the Schrödinger equation.

## Project Overview

In quantum mechanics, wave equations such as the Schrödinger equation describe the behavior of quantum systems. Numerically solving these equations requires bounding the spatial domain. However, due to wave propagation extending beyond these boundaries, it’s important to choose boundary conditions that mitigate unphysical reflections.

This project implements and compares various artificial boundary conditions for the 1D Schrödinger equation, focusing on ABCs and PMLs. The project was supervised by **Brigitte Bidégaray-Fesquet** and **Clément Jourdana**. For more detailed information, refer to the full project report (`report.pdf`).

## Project Structure

```
├── code
│   ├── homogoneous_diffusion_eq
│   │   ├── diffusion_simul_implementation.py
│   │   └── diffusion_test_script.py
│   ├── main_notebook.ipynb  # Main notebook running all simulations
│   ├── nonhomogoneous_diff_eq
│   │   ├── nonhomogoneous_diffusion_simul_implementation.py
│   │   └── nonhomogoneous_diffusion_test_script.py
│   ├── pml
│   │   ├── abc_pml.py  # Implementation of ABC and PML techniques
│   │   ├── abc_pml_test_script.py  # Test scripts
│   │   ├── pml.py
│   │   └── pml_test_script.py
│   ├── schroedinger_eq
│   │   ├── schroedinger_exact_implementation.py
│   │   ├── schroedinger_simul_implementation.py
│   │   └── schroedinger_study_error.py  # Error analysis
│   └── wave_eq
│       ├── wave_exact_implementation.py
│       ├── wave_simul_implementation.py
│       └── wave_study_error.py
├── doc
│   ├── A_Friendly_Review_of_Absorbing_Boundary_Conditions.pdf  # Key reference paper
│   ├── modeling_seminar_bidegaray_jourdana.pdf  # Course statement
│   ├── presentation.pdf  # Presentation slides
│   └── report.pdf  # Project report
├── environment.yml  # Conda environment for dependencies
└── plot  # Visualizations of results and analysis
```

## How to Run the Project

### 1. Setup the Environment

Ensure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed, then set up the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate modeling-seminar
```

### 2. Run Simulations

You can explore the implementations and test cases by running the `main_notebook.ipynb` notebook. This notebook runs simulations for the Schrödinger equation with different boundary conditions and compares the results.

### 3. Visualize Results

The project includes pre-generated plots and code to visualize the results of the simulations. You can find these in the `plot` directory. For example, results for error analysis and boundary condition effectiveness can be found in:

```
plot/schroedinguer/
├── error_abc_only.png
├── sch_eq_abc.png
```

## Goals

The primary objectives of this project are:
- Implement and test Absorbing Boundary Conditions (ABC) and Perfectly Matched Layers (PML).
- Compare the accuracy and efficiency of ABCs and PMLs for absorbing boundary conditions in quantum wave equations.
- Extend the study to other quantum wave equations, such as the Klein-Gordon and Dirac equations.

## References

1. Antoine, X., Lorin, E., & Tang, Q. (2017). A Friendly Review of Absorbing Boundary Conditions and Perfectly Matched Layers for Classical and Relativistic Quantum Wave Equations. *Molecular Physics*, 115(15-16), 1861-1879.
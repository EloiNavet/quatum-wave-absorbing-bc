import numpy as np
import matplotlib.pyplot as plt
from typing import Annotated

plt.style.use("bmh")
plt.rcParams.update({"font.family": "serif"})


# Boundary condition for system Ax = By (x is the unknown)
def apply_boundary_condition(
    matrix_A: np.ndarray,
    matrix_B: np.ndarray,
    cfl_coeff: complex,
    dx: float,
    bctype: str = "dirichlet",
    bcvalues: tuple[float, float] = (0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply boundary condition to the linear system Ax = By.

    Parameters
    ----------
    matrix_A : np.ndarray
        Matrix A of the linear system Ax = By.
    matrix_B : np.ndarray
        Matrix B of the linear system Ax = By.
    cfl_coeff : complex
        CFL coefficient.
    dx : float
        Space step.
    bctype : str, optional
        Type of boundary condition, by default "dirichlet".
    bcvalues : tuple[float, float], optional
        Boundary values, by default (0, 0).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing the modified matrix A, matrix B and vector S.
    """
    assert bctype in [
        "dirichlet",
        "neumann",
    ], "bctype must be one of 'dirichlet', 'neumann'"
    assert np.shape(matrix_A) == np.shape(
        matrix_B
    ), "matrix_A and matrix_B must have the same shape"

    # S is added to fix BC that do not depend on psi
    matrix_S = np.zeros(np.shape(matrix_A)[0], dtype=complex)

    if bctype == "dirichlet":
        # Act as identity on boundary terms (A) and null for B (not dependent on psi)
        matrix_A[0, 0] = matrix_A[-1, -1] = 1
        matrix_B[0, 0] = matrix_B[-1, -1] = 0
        # Add term to have value for dirichlet condition
        matrix_S[0] = bcvalues[0]
        matrix_S[-1] = bcvalues[1]

    elif bctype == "neumann":
        # left
        matrix_A[0, 0] = 1 + cfl_coeff
        matrix_A[0, 1] = -cfl_coeff
        matrix_B[0, 0] = 1 - cfl_coeff
        matrix_B[0, 1] = cfl_coeff
        matrix_S[0] = 2.0 * dx * cfl_coeff * bcvalues[0]
        # right
        matrix_A[-1, -1] = 1 + cfl_coeff
        matrix_A[-1, -2] = -cfl_coeff
        matrix_B[-1, -1] = 1 - cfl_coeff
        matrix_B[-1, -2] = cfl_coeff
        matrix_S[-1] = 2.0 * dx * cfl_coeff * bcvalues[1]

    return matrix_A, matrix_B, matrix_S


def main_experience(
    T: float = 0.1,
    T_nb_point: int = 500,
    space_interval: tuple[float, float] = (-1, 4),
    X_nb_point: int = 500,
    c: float = 1,
    psi_0: np.ndarray | callable = None,
    bctype: str = "neumann",
    bcvalues: tuple[float, float] = (0, 0),
    V: np.ndarray = None,
) -> None:
    """
    Plot diffusion equation.

    Parameters
    ----------
    T : float, optional
        Time simulation duration, by default 0.1.
    T_nb_point : int, optional
        Number of points for time discretization, by default 500.
    space_interval : tuple[float, float], optional
        Space interval, by default (-1, 4).
    X_nb_point : int, optional
        Number of points for space discretization, by default 500.
    c : float, optional
        Space speed, by default 1.
    psi_0 : np.ndarray | callable, optional
        Initial condition, by default None.
    bctype : str, optional
        Type of boundary condition, by default "neumann".
    bcvalues : tuple[float, float], optional
        Boundary value (Neumann & Dirichlet), by default (0, 0).
    V : np.ndarray, optional
        Potential V, by default None.
    """

    # Space mesh
    interval_inf = space_interval[0]
    interval_sup = space_interval[1]
    length_of_interval = interval_sup - interval_inf
    x = np.linspace(0, length_of_interval, X_nb_point)
    dx = length_of_interval / X_nb_point

    # Time mesh
    period_to_emulate = T
    nb_time_points = T_nb_point
    dt = period_to_emulate / nb_time_points

    # Intialisation
    if isinstance(psi_0, np.ndarray):
        psi_previous = psi_0
    elif callable(psi_0):
        psi_previous = psi_0(x)
    else:
        psi_previous = np.exp(
            -(
                (
                    x
                    - (
                        space_interval[0]
                        + 0.3 * (space_interval[1] - space_interval[0])
                    )
                )
                ** 2
            )
            / (0.1**2)
        )
        psi_previous += np.exp(
            -(
                (
                    x
                    - (
                        space_interval[0]
                        + 0.7 * (space_interval[1] - space_interval[0])
                    )
                )
                ** 2
            )
            / (0.1**2)
        )

    if V is None:
        V = np.zeros(X_nb_point)

    # Build linear system to solve Ax = By (where x is the unknown)
    cfl_coeff = c * dt / (dx * dx)
    # Build A
    matrix_A = (1.0 + cfl_coeff) * np.eye(X_nb_point, X_nb_point)
    matrix_A += np.diag([-0.5 * cfl_coeff for i in range(X_nb_point - 1)], 1) + np.diag(
        [-0.5 * cfl_coeff for i in range(X_nb_point - 1)], -1
    )
    matrix_A += np.diag(c * dt * 0.5 * V)  ###############sign ...
    matrix_A[0, :] = matrix_A[-1, :] = 0
    # Build B
    matrix_B = (1.0 - cfl_coeff) * np.eye(X_nb_point, X_nb_point)
    matrix_B += np.diag([0.5 * cfl_coeff for i in range(X_nb_point - 1)], 1) + np.diag(
        [0.5 * cfl_coeff for i in range(X_nb_point - 1)], -1
    )
    matrix_B += np.diag(-c * dt * 0.5 * V)  ###############sign ...
    matrix_B[0, :] = matrix_B[-1, :] = 0
    # Build S
    matrix_A, matrix_B, matrix_S = apply_boundary_condition(
        matrix_A, matrix_B, cfl_coeff, dx, bctype, bcvalues
    )

    # Initialization
    psi = 0.8 * np.exp(-((x - 0.2 * length_of_interval) ** 2) / 0.1**2) + 0.2
    psi_init = psi.copy()

    # Init grapsics
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    fig.show()
    fig.canvas.draw()
    eps = 0.03

    for t_n in range(nb_time_points):
        psi = np.linalg.solve(matrix_A, np.dot(matrix_B, psi) + matrix_S)

        ax.clear()
        ax.plot(x, psi_init, label="Init")
        ax.plot(x, V, "--r", label="Potential")
        ax.plot(x, np.abs(psi), label="Appx")
        ax.set_title("Time {}".format(round(dt * t_n, 6)))
        ax.set_ylim(0 - eps, 1 + eps)
        ax.legend()
        fig.canvas.draw()

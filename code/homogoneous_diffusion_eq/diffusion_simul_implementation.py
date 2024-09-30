import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

plt.style.use("bmh")
plt.rcParams.update({"font.family": "serif"})

BC_TYPE = ["dirichlet", "neumann"]

#############################################################
#                                                           #
#                     BOUNDARY CONDITION                    #
#                                                           #
#############################################################


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
    assert bctype in BC_TYPE, f"bctype must be one of {' '.join(BC_TYPE)}"
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

        matrix_S[0] = 2 * dx * cfl_coeff * bcvalues[0]

        # right
        matrix_A[-1, -1] = 1 + cfl_coeff
        matrix_A[-1, -2] = -cfl_coeff

        matrix_B[-1, -1] = 1 - cfl_coeff
        matrix_B[-1, -2] = cfl_coeff

        matrix_S[-1] = 2 * dx * cfl_coeff * bcvalues[1]

    return matrix_A, matrix_B, matrix_S


#############################################################
#                                                           #
#                     LINEAR SYSTEM BODY                    #
#                                                           #
#############################################################
def build_A_B(
    cfl_coeff: complex,
    X_nb_point: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the matrix A and B for the linear system Ax = By.

    Parameters
    ----------
    cfl_coeff : complex
        CFL coefficient.
    X_nb_point : int
        Number of points in space.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the matrix A and matrix B.
    """
    cfl_diag = 0.5 * np.ones(X_nb_point - 1) * cfl_coeff

    # Build A
    matrix_A = (1 + cfl_coeff) * np.eye(X_nb_point)
    matrix_A += np.diag(-cfl_diag, 1)
    matrix_A += np.diag(-cfl_diag, -1)
    matrix_A[0, :] = matrix_A[-1, :] = 0

    # Build B
    matrix_B = (1 - cfl_coeff) * np.eye(X_nb_point)
    matrix_B += np.diag(cfl_diag, 1)
    matrix_B += np.diag(cfl_diag, -1)
    matrix_B[0, :] = matrix_B[-1, :] = 0

    return matrix_A, matrix_B


#############################################################
#                                                           #
#                            MAIN                           #
#                                                           #
#############################################################
def main_experience(
    period_to_emulate: float = 0.1,
    T_nb_point: int = 500,
    space_interval: tuple[float, float] = (-10, 10),
    X_nb_point: int = 501,
    c: float = 1,
    psi: np.ndarray | callable = None,
    bctype: str = "neumann",
    bcvalues: tuple[float, float] = (0, 0),
    bool_plot: bool = True,
) -> np.ndarray:
    """
    Generate a simulation of the diffusion equation with the given parameters.

    Parameters
    ----------
    period_to_emulate : float, optional
        Time of the simulation, by default 0.1.
    T_nb_point : int, optional
        Number of points in time, by default 500.
    space_interval : tuple[float, float], optional
        Interval of space, by default (-10, 10).
    X_nb_point : int, optional
        Number of points in space, by default 501.
    c : float, optional
        Speed of diffusion, by default 1.
    psi : np.ndarray | callable, optional
        Initial condition, by default None.
    bctype : str, optional
        Type of boundary condition, by default "neumann".
    bcvalues : tuple[float, float], optional
        Boundary values, by default (0, 0).
    bool_plot : bool, optional
        Whether to plot the solution, by default True.

    Returns
    -------
    np.ndarray
        Solution of the diffusion equation.
    """
    print("Computing the solution of the diffusion equation...")

    # Space mesh
    interval_inf, interval_sup = space_interval
    x = np.linspace(interval_inf, interval_sup, X_nb_point)
    dx = (interval_sup - interval_inf) / (X_nb_point - 1)

    # Time mesh
    t = np.linspace(0, period_to_emulate, T_nb_point)
    dt = period_to_emulate / (T_nb_point - 1)

    # Intialisation
    if isinstance(psi, np.ndarray):
        psi = psi
    elif callable(psi):
        psi = psi(x)
    else:
        psi = np.exp(
            -((x - (interval_inf + 0.3 * (interval_sup - interval_inf))) ** 2)
            / (0.1**2)
        )
        psi += np.exp(
            -((x - (interval_inf + 0.7 * (interval_sup - interval_inf))) ** 2)
            / (0.1**2)
        )

    cfl_coeff = c * dt / (dx * dx)

    # Build A and B
    matrix_A, matrix_B = build_A_B(cfl_coeff, X_nb_point)

    # Build S
    matrix_A, matrix_B, matrix_S = apply_boundary_condition(
        matrix_A, matrix_B, cfl_coeff, dx, bctype, bcvalues
    )

    # Plot the solution
    psis = np.zeros((T_nb_point, X_nb_point), dtype=complex)
    psis[0] = psi

    for t_n in tqdm(range(1, T_nb_point), desc="Progress", unit="step"):
        psis[t_n] = np.linalg.solve(
            matrix_A, np.dot(matrix_B, psis[t_n - 1]) + matrix_S
        )

        if t_n == 1:
            print(
                "\tCondition number for A : {}".format(
                    np.linalg.norm(matrix_A) * np.linalg.norm(np.linalg.inv(matrix_A))
                )
            )

    if bool_plot:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        psm = axs[0].pcolormesh(
            x,
            t,
            np.abs(psis),
            cmap="turbo",
        )
        fig.colorbar(psm, ax=axs[0])
        axs[0].grid(False)
        axs[0].set_xlabel("Space")
        axs[0].set_ylabel("Time")
        axs[0].set_title(r"Heatmap of $\psi$" + f" with {bctype} bc.")

        # plot energy evolution
        def compute_L2_norm(psis):
            return np.array([np.linalg.norm(psi) for psi in psis])

        initial_energy = np.linalg.norm(psis[0])
        L2_energy = compute_L2_norm(psis)
        axs[1].plot(L2_energy, t, label=r"$||\psi(\cdot,t)||_{L_2}$")
        # axes[1].plot(1/np.sqrt(4*np.pi*c*t)*initial_energy, t, label = r"$||\psi(\cdot,0)||_{L_2}/\sqrt{4c\pi t}$", linestyle = 'dotted')
        axs[1].plot(
            [initial_energy] * T_nb_point,
            t,
            label=r"$||\psi(\cdot,0)||_{L_2}$",
            linestyle="--",
        )
        axs[1].set_ylim([0, period_to_emulate])
        axs[1].set_xlim([-0.001, 1.1 * np.max(L2_energy)])
        axs[1].set_xlabel("Energy")
        axs[1].set_title(r"Energy variation, i.e. $||\psi||_{L_2}(t)$")
        axs[1].legend()

        fig.tight_layout()
        plt.show()

    return psis

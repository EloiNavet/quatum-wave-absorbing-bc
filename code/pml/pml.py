import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

plt.style.use("bmh")
plt.rcParams.update({"font.family": "serif"})


#############################################################
#                                                           #
#                         UTILITARIES                       #
#                                                           #
#############################################################
PHASE_FACTOR = np.exp(0.25j * np.pi)


def function_S(x: float, L_inner: float, sigma_0: float) -> float:
    """
    Function S for the PML. It is a function of x, L_inner and sigma_0.

    Parameters
    ----------
    x : float
        Position.
    L_inner : float
        Inner limit of the PML.
    sigma_0 : float
        Parameter of the PML.

    Returns
    -------
    float
        Value of the function S(x).
    """
    sigma = sigma_0 * (np.abs(x) - L_inner) ** 2
    return 1 + PHASE_FACTOR * sigma


#############################################################
#                                                           #
#                     BOUNDARY CONDITION                    #
#                                                           #
#############################################################
def apply_boundary_condition(
    psi_left: np.ndarray,
    psi_right: np.ndarray,
    length: int,
    matrix_A: np.ndarray,
    matrix_B: np.ndarray,
    bc_right_left_value=[0, 0],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply boundary condition to the linear system Ax = By.

    Parameters
    ----------
    psi_left : np.ndarray
        Left boundary condition.
    psi_right : np.ndarray
        Right boundary condition.
    length : int
        Length of the system.
    matrix_A : np.ndarray
        Matrix A of the linear system Ax = By.
    matrix_B : np.ndarray
        Matrix B of the linear system Ax = By.
    bc_right_left_value : list[float], optional
        Boundary values, by default [0, 0].

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing the modified matrix A, matrix B and vector S.
    """
    assert len(psi_right) == len(
        psi_left
    ), "psi_left, psi_right and betas must have the same length"

    vector_S = np.zeros(length, dtype=complex)

    # Matrix A
    matrix_A[0, 0] = 1
    matrix_A[-1, -1] = 1
    # Second member
    vector_S[0] = bc_right_left_value[0]
    vector_S[-1] = bc_right_left_value[1]

    return matrix_A, matrix_B, vector_S


#############################################################
#                                                           #
#                     LINEAR SYSTEM BODY                    #
#                                                           #
#############################################################
def build_A_B(
    c: complex,
    dt: float,
    dx: float,
    X_nb_point: int,
    PML: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the matrix A and B for the linear system Ax = By.

    Parameters
    ----------
    c : complex
        Speed of the wave.
    dt : float
        Time step.
    dx : float
        Space step.
    X_nb_point : int
        Number of points in space.
    PML : np.ndarray
        PML matrix of size X_nb_point.
    """
    cfl_coeff = c * dt / (dx * dx)
    cfl_upper_diag = 0.5 * np.ones(X_nb_point - 1) * cfl_coeff * PML[1:]
    cfl_lower_diag = 0.5 * np.ones(X_nb_point - 1) * cfl_coeff * PML[:-1]

    # Build A
    matrix_A = (1 + cfl_coeff * PML.T) * np.eye(X_nb_point, X_nb_point)
    matrix_A += np.diag(-cfl_upper_diag, 1)
    matrix_A += np.diag(-cfl_lower_diag, -1)
    matrix_A[0, :] = 0.0
    matrix_A[-1, :] = 0.0

    # Build B
    matrix_B = (1 - cfl_coeff * PML.T) * np.eye(X_nb_point, X_nb_point)
    matrix_B += np.diag(cfl_upper_diag, 1)
    matrix_B += np.diag(cfl_lower_diag, -1)
    matrix_B[0, :] = 0.0
    matrix_B[-1, :] = 0.0

    return matrix_A, matrix_B


#############################################################
#                                                           #
#                            MAIN                           #
#                                                           #
#############################################################
def main_experience(
    period_to_emulate: float = 0.5,
    T_nb_point: int = 500,
    space_interval: tuple[float, float] = (-15, 15),
    X_nb_point: int = 501,
    psi: np.ndarray = None,
    delta: tuple[float, float] = (5.0, 5.0),
    sigma_0: float = 0.05,
    bool_plot: bool = True,
) -> np.ndarray:
    """
    Computes the solution of the Schrodinger equation with PML.
    Note: the simulation is made over the whole interval space_interval, and delta is substrated from the interval.

    Parameters
    ----------
    period_to_emulate : float, optional
        Period of time to emulate, by default 0.5.
    T_nb_point : int, optional
        Number of points in time, by default 500.
    space_interval : tuple[float, float], optional
        Interval of space, by default (-15, 15).
    X_nb_point : int, optional
        Number of points in space, by default 501.
    psi : np.ndarray, optional
        Initial condition, by default None.
    delta : tuple(float, float), optional
        Size of the PML, by default (5.0, 5.0).
    sigma_0 : float, optional
        Parameter of the PML, by default 0.05.
    bool_plot : bool, optional
        Whether to plot the solution or not, by default True.

    Returns
    -------
    np.ndarray
        Solution of the Schrodinger equation with PML.
    """
    assert (
        space_interval[0] < space_interval[1]
    ), "space_interval must be a tuple (a, b) with a < b"
    assert (
        delta[0] + delta[1] < space_interval[1] - space_interval[0]
    ), "delta must be a tuple (a, b) with a + b < b - a"

    print("Starting the simulation of the Schrodinger equation with PML...")

    # Space mesh
    interval_inf, interval_sup = space_interval
    x = np.linspace(interval_inf, interval_sup, X_nb_point)
    dx = (interval_sup - interval_inf) / (X_nb_point - 1)
    inner_index_left = int(delta[0] / dx)
    inner_index_right = int(delta[1] / dx)

    # Time mesh
    t = np.linspace(0, period_to_emulate, T_nb_point)
    dt = period_to_emulate / (T_nb_point - 1)

    # Precomputing the S(x)
    PML_matrix = np.ones(X_nb_point, dtype=complex)

    PML_matrix[:inner_index_left] = (
        1 / (function_S(x[:inner_index_left], interval_inf - delta[0], sigma_0)) ** 2
    )
    PML_matrix[-inner_index_right:] = (
        1 / (function_S(x[-inner_index_right:], interval_sup - delta[1], sigma_0)) ** 2
    )

    # Build linear system to solve Ax = By (where x is the unknown)
    C = 1j
    matrix_A, matrix_B = build_A_B(C, dt, dx, X_nb_point, PML_matrix)

    # Initialization
    if isinstance(psi, np.ndarray):
        pass
    elif callable(psi):
        psi = psi(x)
    else:
        psi = 2 / np.cosh(np.sqrt(2) * x) * np.exp(1j * 15 / 2 * x)  # See the article

    psis = np.zeros((T_nb_point, X_nb_point), dtype=complex)
    psis[0] = psi

    # Timeloop with linear system to solve
    for t_n in tqdm(range(1, T_nb_point), desc="Progress", unit="step"):

        matrix_A, matrix_B, vector_S = apply_boundary_condition(
            psis[: t_n - 1, 0], psis[: t_n - 1, -1], len(x), matrix_A, matrix_B
        )

        psis[t_n] = np.linalg.solve(
            matrix_A, np.dot(matrix_B, psis[t_n - 1]) + vector_S
        )

        if t_n == 1:
            print(
                "\tCondition number for A : {}".format(
                    np.linalg.norm(matrix_A) * np.linalg.norm(np.linalg.inv(matrix_A))
                )
            )

    # Visualization
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
        axs[0].set_title(r"Heatmap of $|\psi|$")
        axs[0].axvline(x=interval_sup - delta[1], color="r")
        axs[0].axvline(x=interval_inf + delta[0], color="r")

        # Plot evolution for L1 energy
        def compute_L2_norm(psis):
            return [np.linalg.norm(psi) for psi in psis]

        initial_energy = np.linalg.norm(psi[inner_index_right:-inner_index_right])
        L2_energy = compute_L2_norm(psis)
        axs[1].plot(L2_energy, t, label=r"$||\psi(\cdot,t)||_{L_2}$")
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


if __name__ == "__main__":
    main_experience(period_to_emulate=2)

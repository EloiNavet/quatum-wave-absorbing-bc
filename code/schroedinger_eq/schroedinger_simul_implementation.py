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
PHASE = np.exp(-0.25j * np.pi)
PHASE_CONJ = np.exp(0.25j * np.pi)


def gamma(k: int) -> float:
    """
    Gamma function. See the report for more details.

    Parameters
    ----------
    k : int
        Integer.

    Returns
    -------
    float
        Value of the gamma function at k.
    """
    if k <= 0:
        return 1
    if k == 1:
        return 0
    return (k - 1) * gamma(k - 2) / k


def alpha(k: int) -> float:
    """
    Alpha function. See the report for more details.

    Parameters
    ----------
    k : int
        Integer.

    Returns
    -------
    float
        Value of the alpha function at k.
    """
    assert k >= 0, "k must be a positive integer"
    if k in [0, 1]:
        return 1
    return gamma(k) + gamma(k - 1)


def beta(k: int) -> float:
    """
    Beta function. See the report for more details.

    Parameters
    ----------
    k : int
        Integer.

    Returns
    -------
    float
        Value of the beta function at k.
    """
    assert k >= 0, "k must be a positive integer"
    return (-1) ** k * alpha(k)


def function_S(x: float, L_inner: float, sigma_0: float) -> float:
    """
    Function S of the PML. See the report for more details.

    Parameters
    ----------
    x : float
        Value of x.
    L_inner : float
        Inner value of L.
    sigma_0 : float
        Parameter of the PML.

    Returns
    -------
    float
        Value of S at x.
    """
    sigma = sigma_0 * (np.abs(x) - L_inner) ** 2
    return 1 + PHASE_CONJ * sigma


#############################################################
#                                                           #
#                     BOUNDARY CONDITION                    #
#                                                           #
#############################################################
def apply_boundary_condition(
    psi_left: np.ndarray,
    psi_right: np.ndarray,
    matrix_A: np.ndarray,
    matrix_B: np.ndarray,
    length: int,
    factor: float = None,
    dx: float = None,
    dt: float = None,
    betas: np.ndarray = None,
    bctype: str = "transparent",
    bc_right_left_value=[0, 0],
    V_independent_integrals=None,
    V_independent_normal_derivatives=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the boundary condition to the matrix A and B.

    Parameters
    ----------
    psi_left : np.ndarray
        Psi left.
    psi_right : np.ndarray
        Psi right.
    matrix_A : np.ndarray
        Matrix A of the linear system.
    matrix_B : np.ndarray
        Matrix B of the linear system.
    length : int
        Length of the space mesh.
    factor : float, optional
        Factor, by default None.
    dx : float, optional
        Space step, by default None.
    dt : float, optional
        Time step, by default None.
    betas : np.ndarray, optional
        Betas, by default None.
    bctype : str, optional
        Boundary condition type, by default "transparent".
    bc_right_left_value : list[float], optional
        Boundary condition values, by default [0, 0].
    V_independent_integrals : np.ndarray, optional
        Integral over time of the 'linear' part of the potential, by default None.
    V_independent_normal_derivatives : np.ndarray, optional
        Normal derivatives of the potential, by default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Matrix A, matrix B and vector S.
    """

    assert bctype in [
        "transparent_basic",
        "transparent_potential",
        "dirichlet",
        "neumann",
    ], "bctype must be one of 'transparent_basic', 'dirichlet', 'neumann', 'transparent_potential'"

    assert (
        len(psi_right) == len(psi_left) == len(betas) - 2
    ), "psi_left, psi_right and betas\{0,1} must have the same length"
    vector_S = np.zeros(length, dtype=complex)

    if bctype == "transparent_basic":
        # Matrix A
        matrix_A[0, 0] = 1.5 + factor * dx * betas[0]
        matrix_A[0, 1] = -2
        matrix_A[0, 2] = 0.5
        matrix_A[-1, -1] = 1.5 + factor * dx * betas[0]
        matrix_A[-1, -2] = -2
        matrix_A[-1, -3] = 0.5
        # MAtrix B
        matrix_B[0, 0] = -factor * dx * betas[1]
        matrix_B[0, 1] = 0
        matrix_B[-1, -1] = -factor * dx * betas[1]
        matrix_B[-1, -2] = 0

        betas = betas[2:]
        # Time memory term
        vector_S[0] = -dx * factor * np.sum(betas * psi_left[::-1])
        vector_S[-1] = -dx * factor * np.sum(betas * psi_right[::-1])

    elif bctype == "transparent_potential":

        # Matrix A
        matrix_A[0, 0] = 1.5 + factor * dx * betas[0]
        matrix_A[0, 1] = -2
        matrix_A[0, 2] = 0.5
        matrix_A[-1, -1] = 1.5 + factor * dx * betas[0]
        matrix_A[-1, -2] = -2
        matrix_A[-1, -3] = 0.5
        if list(V_independent_normal_derivatives) != []:
            dnV_L = V_independent_normal_derivatives[-1, 0]
            dnV_R = V_independent_normal_derivatives[-1, -1]
            matrix_A[0, 0] += (
                dx
                * 1.0j
                * np.sign(dnV_L)
                * 0.25
                * (np.abs(dnV_L) ** 2)
                * 0.5
                * dt
                * np.exp(1.0j * V_independent_integrals[-1, 0])
            )
            matrix_A[-1, -1] += (
                dx
                * 1.0j
                * np.sign(dnV_R)
                * 0.25
                * (np.abs(dnV_R) ** 2)
                * 0.5
                * dt
                * np.exp(1.0j * V_independent_integrals[-1, -1])
            )

        # Matrix B
        matrix_B[0, 0] = -factor * dx * betas[1]
        matrix_B[0, 1] = 0
        matrix_B[-1, -1] = -factor * dx * betas[1]
        matrix_B[-1, -2] = 0
        if V_independent_normal_derivatives.size > 2:
            tmp_L = np.exp(1.0j * V_independent_integrals[-1, 0]) * np.exp(
                -1.0j * V_independent_integrals[-2, 0]
            )
            tmp_R = np.exp(1.0j * V_independent_integrals[-1, -1]) * np.exp(
                -1.0j * V_independent_integrals[-2, -1]
            )
            dnV_L = V_independent_normal_derivatives[-1, 0]
            dnV_R = V_independent_normal_derivatives[-1, -1]
            matrix_B[0, 0] *= tmp_L
            matrix_B[0, 0] -= (
                1.0j
                * np.sign(dnV_L)
                * tmp_L
                * 0.25
                * np.sqrt(np.abs(dnV_L * V_independent_normal_derivatives[-2, 0]))
                * dt
                * dx
            )
            matrix_B[-1, -1] *= tmp_R
            matrix_B[-1, -1] -= (
                1.0j
                * np.sign(dnV_R)
                * tmp_L
                * 0.25
                * np.sqrt(np.abs(dnV_R * V_independent_normal_derivatives[-2, -1]))
                * dt
                * dx
            )

        betas_tmp = betas[2:]
        # Time memory term
        vector_S[0] = -dx * factor * np.sum(betas_tmp * psi_left[::-1])
        vector_S[-1] = -dx * factor * np.sum(betas_tmp * psi_right[::-1])
        if (
            psi_left.size >= 2 and len(betas_tmp) >= 2
        ):  # S only has sense if t_n -2 is non negative index.
            dnV_L = V_independent_normal_derivatives[-1, 0]
            dnV_R = V_independent_normal_derivatives[-1, -1]
            vector_S[0] = (
                -dx
                * factor
                * np.exp(1.0j * V_independent_integrals[0, -1])
                * np.sum(
                    np.exp(-1.0j * (V_independent_integrals[:-2, 0])[::-1])
                    * betas_tmp
                    * psi_left[::-1]
                )
            )
            vector_S[0] -= (
                dx
                * 1.0j
                * np.sign(dnV_L)
                * 0.5
                * np.sqrt(np.abs(dnV_L))
                * np.exp(1.0j * V_independent_integrals[-1, 0])
                * dt
                * np.sum(
                    0.5
                    * np.sqrt(np.abs(V_independent_normal_derivatives[1:-2, 0]))
                    * np.exp(-1.0j * V_independent_integrals[1:-2, 0])
                    * psi_left[1:]
                )
            )
            vector_S[0] -= (
                dx
                * 1.0j
                * np.sign(dnV_L)
                * 0.5
                * np.sqrt(np.abs(dnV_L))
                * np.exp(1.0j * V_independent_integrals[-1, 0])
                * 0.25
                * dt
                * np.sqrt(np.abs(V_independent_normal_derivatives[0, 0]))
                * np.exp(-1.0j * V_independent_integrals[0, 0])
                * psi_left[0]
            )

            vector_S[-1] = (
                -dx
                * factor
                * np.exp(1.0j * V_independent_integrals[-1, -1])
                * np.sum(
                    np.exp(-1.0j * (V_independent_integrals[:-2, -1])[::-1])
                    * betas_tmp
                    * psi_right[::-1]
                )
            )
            vector_S[-1] -= (
                dx
                * 1.0j
                * np.sign(dnV_R)
                * 0.5
                * np.sqrt(np.abs(dnV_R))
                * np.exp(1.0j * V_independent_integrals[-1, -1])
                * dt
                * np.sum(
                    0.5
                    * np.sqrt(np.abs(V_independent_normal_derivatives[1:-2, -1]))
                    * np.exp(-1.0j * V_independent_integrals[1:-2, -1])
                    * psi_right[1:]
                )
            )
            vector_S[-1] -= (
                dx
                * 1.0j
                * np.sign(dnV_R)
                * 0.5
                * np.sqrt(np.abs(dnV_R))
                * np.exp(1.0j * V_independent_integrals[-1, -1])
                * 0.25
                * dt
                * np.sqrt(np.abs(V_independent_normal_derivatives[0, -1]))
                * np.exp(-1.0j * V_independent_integrals[0, -1])
                * psi_right[0]
            )

    elif bctype == "dirichlet":
        # Matrix A
        matrix_A[0, 0] = 1.0
        matrix_A[-1, -1] = 1.0

        # Second member
        vector_S[0] = bc_right_left_value[0]
        vector_S[-1] = bc_right_left_value[1]

    elif bctype == "neumann":
        # Matrix A
        matrix_A[0, 0] = 3
        matrix_A[0, 1] = -4
        matrix_A[0, 2] = 1
        matrix_A[-1, -1] = 3
        matrix_A[-1, -2] = -4
        matrix_A[-1, -3] = 1
        # Second member
        vector_S[0] = 2.0 * dx * bc_right_left_value[0]
        vector_S[-1] = 2.0 * dx * bc_right_left_value[1]

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
    V_independent: np.ndarray,
    activate_PML: bool,
    PML: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the matrix A and B of the linear system.

    Parameters
    ----------
    c : complex
        Speed of the wave.
    dt : float
        Time step.
    dx : float
        Space step.
    X_nb_point : int
        Number of points in the space mesh.
    V_independent : np.ndarray
        Potential independent of the wave.
    activate_PML : bool
        Whether to activate the PML or not.
    PML : np.ndarray, optional
        PML, by default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Matrix A and matrix B.
    """

    if activate_PML:
        cfl_coeff = c * dt / (dx * dx)
        cfl_upper_diag = 0.5 * np.ones(X_nb_point - 1) * cfl_coeff * PML[1:]
        cfl_lower_diag = 0.5 * np.ones(X_nb_point - 1) * cfl_coeff * PML[:-1]

        # Build A
        matrix_A = (1 + cfl_coeff * PML.T) * np.eye(X_nb_point)
        matrix_A -= np.diag(
            0.5 * c * dt * V_independent[1, :]
        )  # space dependent potential (does not depend on psi)
        matrix_A += np.diag(-cfl_upper_diag, 1)
        matrix_A += np.diag(-cfl_lower_diag, -1)
        matrix_A[0, :] = 0.0
        matrix_A[-1, :] = 0.0

        # Build B
        matrix_B = (1 - cfl_coeff * PML.T) * np.eye(X_nb_point)
        matrix_B += np.diag(
            0.5 * c * dt * V_independent[0, :]
        )  # space dependent potential (does not depend on psi)
        matrix_B += np.diag(cfl_upper_diag, 1)
        matrix_B += np.diag(cfl_lower_diag, -1)
        matrix_B[0, :] = 0.0
        matrix_B[-1, :] = 0.0

        return matrix_A, matrix_B

    cfl_coeff = c * dt / (dx * dx)
    cfl_diag = 0.5 * np.ones(X_nb_point - 1) * cfl_coeff

    # Build A
    matrix_A = (1 + cfl_coeff) * np.eye(X_nb_point, X_nb_point)
    matrix_A -= np.diag(
        0.5 * c * dt * V_independent[1, :]
    )  # space dependent potential (does not depend on psi)
    matrix_A += np.diag(-cfl_diag, 1)
    matrix_A += np.diag(-cfl_diag, -1)
    matrix_A[0, :] = 0.0
    matrix_A[-1, :] = 0.0

    # Build B
    matrix_B = (1 - cfl_coeff) * np.eye(X_nb_point, X_nb_point)
    matrix_B += np.diag(
        0.5 * c * dt * V_independent[0, :]
    )  # space dependent potential (does not depend on psi)
    matrix_B += np.diag(cfl_diag, 1)
    matrix_B += np.diag(cfl_diag, -1)
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
    T_nb_point: int = 501,
    space_interval: tuple[float, float] = (-15, 15),
    X_nb_point: int = 501,
    psi: np.ndarray | callable = None,
    bctype: str = "transparent",
    bc_right_left_value: tuple[float, float] = [0, 0],
    activate_PML: bool = False,
    delta: tuple[float, float] = (5, 5),
    sigma_0: float = 0.05,
    V_independent: np.ndarray = None,
    V_non_linear_part: np.ndarray | callable = None,
    bool_plot: bool = True,
) -> np.ndarray:
    """
    Makes the simulation of Schrodinger equation.

    Parameters
    ----------
    period_to_emulate : float, optional
        Period to emulate, by default 0.5.
    T_nb_point : int, optional
        Number of points in the time mesh, by default 501.
    space_interval : tuple[float, float], optional
        Length of the space interval, by default (-15, 15).
    X_nb_point : int, optional
        Number of points in the space mesh, by default 501.
    psi : np.ndarray | callable, optional
        Initial condition, by default None.
    bctype : str, optional
        Boundary condition type, by default "transparent".
    bc_right_left_value : tuple[float, float], optional
        Boundary condition values, by default [0, 0].
    activate_PML : bool, optional
        Specifiate PML or not, by default False.
    delta : tuple[float, float], optional
        Size of the PML (included in space_interval), by default (5, 5).
    sigma_0 : float, optional
        Parameter of the PML, by default 0.05.
    V_independent : np.ndarray, optional
        Potential, by default None.
    V_non_linear_part : np.ndarray | callable, optional
        Non linear part of the potential, by default None.
    bool_plot : bool, optional
        Boolean to plot the result, by default True.

    Returns
    -------
    np.ndarray
        Result of the simulation.
    """
    print(f"Starting the simulation of the Schrodinger equation with {bctype}...")

    # Space mesh
    interval_inf, interval_sup = space_interval
    x = np.linspace(interval_inf, interval_sup, X_nb_point)
    dx = (interval_sup - interval_inf) / (X_nb_point - 1)

    # Time mesh
    t = np.linspace(0, period_to_emulate, T_nb_point)
    dt = period_to_emulate / (T_nb_point - 1)

    # Necessary data for PML
    PML_matrix = None
    if activate_PML:
        assert (
            space_interval[0] < space_interval[1]
        ), "space_interval must be a tuple (a, b) with a < b"
        assert (
            delta[0] + delta[1] < space_interval[1] - space_interval[0]
        ), "delta must be a tuple (a, b) with a + b < b - a"
        inner_index_left = int(delta[0] / dx)
        inner_index_right = int(delta[1] / dx)
        # Precomputing the S(x)
        PML_matrix = np.ones(X_nb_point, dtype=complex)

        PML_matrix[:inner_index_left] = (
            1
            / (function_S(x[:inner_index_left], interval_inf - delta[0], sigma_0)) ** 2
        )
        PML_matrix[-inner_index_right:] = (
            1
            / (function_S(x[-inner_index_right:], interval_sup - delta[1], sigma_0))
            ** 2
        )

    # Useful data
    c = 1j
    factor = PHASE * np.sqrt(2 / dt)

    # Source and speed terms
    potential_is_time_dpt = True
    if isinstance(
        V_independent, np.ndarray
    ):  # assume only space dependent here so duplicate over time
        if isinstance(V_independent, np.ndarray):
            V_independent = np.broadcast_to(V_independent, (T_nb_point, X_nb_point))
            potential_is_time_dpt = False
    elif callable(V_independent):
        test_for_time_dpt_1 = [
            V_independent(0.25 * (interval_inf + interval_sup), t_point)
            for t_point in [0.0, 0.01, 0.1, 1.0]
        ]
        test_for_time_dpt_2 = [
            V_independent(0.75 * (interval_inf + interval_sup), t_point)
            for t_point in [0.0, 0.01, 0.1, 1.0]
        ]
        potential_is_time_dpt = (
            test_for_time_dpt_1[0] == test_for_time_dpt_1[1]
            and test_for_time_dpt_1[0] == test_for_time_dpt_1[2]
            and test_for_time_dpt_1[0] == test_for_time_dpt_1[3]
        )
        potential_is_time_dpt = not (
            potential_is_time_dpt
            and (
                test_for_time_dpt_2[0] == test_for_time_dpt_2[1]
                and test_for_time_dpt_2[0] == test_for_time_dpt_2[2]
                and test_for_time_dpt_2[0] == test_for_time_dpt_2[3]
            )
        )
        V_independent = np.array([V_independent(x, t_point) for t_point in t])
    else:
        V_independent = np.zeros((T_nb_point, X_nb_point))
        potential_is_time_dpt = False

    print("\tPotential is time dpt :", potential_is_time_dpt)

    ########################################### to be continued
    if isinstance(V_non_linear_part, np.ndarray):
        pass  # the potential is already given
    elif callable(V_non_linear_part):
        V_non_linear_part = V_non_linear_part(x)
    else:
        V_non_linear_part = np.zeros(X_nb_point)
    ########################################### to be continued

    # Precomputing values of interest
    # Beta coeffs
    betas = [beta(k) for k in range(T_nb_point + 1)]
    # integral over time of the 'linear' part of the potential (trapezoid method)
    V_independent_integrals = np.transpose(
        np.array(
            [
                np.cumsum(
                    0.5
                    * dt
                    * (V_independent[:, 0] + np.roll(V_independent[:, 0], shift=-1))
                ),
                np.cumsum(
                    0.5
                    * dt
                    * (V_independent[:, -1] + np.roll(V_independent[:, -1], shift=-1))
                ),
            ]
        )
    )

    V_independent_normal_derivatives = np.zeros((T_nb_point, 2))

    V_independent_normal_derivatives[:-1, 0] = (
        3.0 * V_independent[:-1, 0]
        - 4.0 * V_independent[:-1, 1]
        + V_independent[:-1, 2]
    ) / (2.0 * dx)
    V_independent_normal_derivatives[:-1, -1] = (
        3.0 * V_independent[:-1, -1]
        - 4.0 * V_independent[:-1, -2]
        + V_independent[:-1, -3]
    ) / (2.0 * dx)

    # Initialization
    if isinstance(psi, np.ndarray):
        pass
    elif callable(psi):
        psi = psi(x)
    else:
        psi = 2 / np.cosh(np.sqrt(2) * x) * np.exp(1j * 15 / 2 * x)  # See the article

    # keep psi_0 in memeory (needed for L2 energy computation)
    psis = np.zeros((T_nb_point, X_nb_point), dtype=complex)
    psis[0] = psi

    # Timeloop with linear system to solve
    for t_n in tqdm(range(1, T_nb_point), desc="Progress", unit="step"):

        # time dpt potential
        if t_n == 1 or potential_is_time_dpt:
            matrix_A, matrix_B = build_A_B(
                c,
                dt,
                dx,
                X_nb_point,
                V_independent[t_n - 1 : t_n + 1, :],
                activate_PML,
                PML_matrix,
            )

        matrix_A, matrix_B, vector_S = apply_boundary_condition(
            psi_left=psis[: t_n - 1, 0],
            psi_right=psis[: t_n - 1, -1],
            matrix_A=matrix_A,
            matrix_B=matrix_B,
            length=len(x),
            factor=factor,
            dx=dx,
            dt=dt,
            betas=betas[: t_n + 1],
            bctype=bctype,
            bc_right_left_value=bc_right_left_value,
            V_independent_integrals=V_independent_integrals[: t_n + 1, :],
            V_independent_normal_derivatives=V_independent_normal_derivatives[
                : t_n + 1, :
            ],
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
        plot_result(
            psis=psis,
            psi_0=psi,
            period_to_emulate=period_to_emulate,
            T_nb_point=T_nb_point,
            space_interval=space_interval,
            X_nb_point=X_nb_point,
            bctype=bctype,
            activate_PML=activate_PML,
            delta=delta,
        )

    return psis


#############################################################
#                                                           #
#                           UTILS                           #
#                                                           #
#############################################################


def plot_result(
    psis,
    psi_0,
    period_to_emulate: float,
    T_nb_point: int,
    space_interval: tuple[float, float],
    X_nb_point: int,
    bctype: str,
    activate_PML: bool = False,
    delta: tuple[float, float] = (5, 5),
):
    """
    Plot the result of the simulation.

    Parameters
    ----------
    psis : np.ndarray
        Result of the simulation.
    psi_0 : np.ndarray
        Initial condition.
    period_to_emulate : float
        Period to emulate.
    T_nb_point : int
        Number of points in the time mesh.
    space_interval : tuple[float, float]
        Space interval.
    X_nb_point : int
        Number of points in the space mesh.
    bctype : str
        Boundary condition type.
    activate_PML : bool, optional
        Whether to activate the PML or not, by default False.
    delta : tuple[float, float], optional
        Size of the PML, by default (5, 5).
    """

    def compute_L2_norm(psis):
        return np.array([np.linalg.norm(psi) for psi in psis])

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Space mesh
    interval_inf, interval_sup = space_interval
    x = np.linspace(interval_inf, interval_sup, X_nb_point)
    dx = (interval_sup - interval_inf) / (X_nb_point - 1)

    # Time mesh
    t = np.linspace(0, period_to_emulate, T_nb_point)

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
    axs[0].set_title(r"Heatmap of $|\psi_{\text{simul}}|$")
    axs[0].axvline(x=interval_inf, color="b")
    axs[0].axvline(x=interval_sup, color="b")

    initial_energy = np.linalg.norm(psi_0)

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
    axs[1].set_title(r"$L_2$ Energy, i.e. $||\psi||_{L_2}(t)$")
    axs[1].legend()

    fig.suptitle(
        "Schrodinger equation simulation with "
        + bctype
        + " BC"
        + activate_PML * " and with PML",
        fontsize=16,
    )

    if activate_PML:
        inner_index_left = int(delta[0] / dx)
        inner_index_right = int(delta[1] / dx)

        axs[0].axvline(x=interval_inf + delta[0], color="r")
        axs[0].axvline(x=interval_sup - delta[1], color="r")

        fig.tight_layout()
        plt.show()

        # zoomed plot
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        psis_inner = psis[:, inner_index_left:-inner_index_right]
        x_inner = x[inner_index_left:-inner_index_right]

        psm = axs[0].pcolormesh(
            x_inner,
            t,
            np.abs(psis_inner),
            cmap="turbo",
        )
        fig.colorbar(psm, ax=axs[0])
        axs[0].grid(False)
        axs[0].set_xlabel("Space")
        axs[0].set_ylabel("Time")
        axs[0].set_title(r"Heatmap of $|\psi^{\text{inner}}_{\text{simul}}|$")

        initial_energy = initial_energy = np.linalg.norm(
            psi_0[inner_index_left:-inner_index_right]
        )

        L2_energy = compute_L2_norm(psis_inner)
        axs[1].plot(L2_energy, t, label=r"$||\psi^{\text{inner}}(\cdot,t)||_{L_2}$")
        axs[1].plot(
            [initial_energy] * T_nb_point,
            t,
            label=r"$||\psi^{\text{inner}}(\cdot,0)||_{L_2}$",
            linestyle="--",
        )
        axs[1].set_ylim([0, period_to_emulate])
        axs[1].set_xlim([-0.001, 1.1 * np.max(L2_energy)])
        axs[1].set_xlabel("Energy")
        axs[1].set_title(r"Energy variation, i.e. $||\psi^{\text{inner}}||_{L_2}(t)$")
        axs[1].legend()
        fig.suptitle(
            "Schrodinger equation simulation with "
            + bctype
            + " BC and with PML. Focused on the area of interest",
            fontsize=16,
        )

    fig.tight_layout()
    plt.show()

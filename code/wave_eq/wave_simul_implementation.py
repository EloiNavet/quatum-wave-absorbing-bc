import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

plt.style.use("bmh")
plt.rcParams.update({"font.family": "serif"})

BC_TYPE = ["dirichlet", "periodic", "transparent", "neumann"]


#############################################################
#                                                           #
#                     BOUNDARY CONDITION                    #
#                                                           #
#############################################################
def apply_boundary_condition(
    psi_next: np.ndarray,
    psi_current: np.ndarray,
    psi_previous: np.ndarray,
    c: float,
    dt: float,
    dx: float,
    Q: np.ndarray,
    bctype: str = "reflective",
    bc_right_left_value: tuple[float, float] = None,
) -> np.ndarray:
    """
    Compute psi_next with the boundary condition.

    Parameters
    ----------
    psi_next : np.ndarray
        Psi at the next time step.
    psi_current : np.ndarray
        Psi at the current time step.
    psi_previous : np.ndarray
        Psi at the previous time step.
    c : float
        Speed of the wave.
    dt : float
        Time step.
    dx : float
        Space step.
    Q : np.ndarray
        Source term.
    bctype : str, optional
        Boundary condition type, by default "reflective".
    bc_right_left_value : tuple[float, float], optional
        Boundary condition value, by default None.

    Returns
    -------
    np.ndarray
        Psi at the next time step.
    """
    assert bctype in BC_TYPE, f"bctype must be one of {' ' .join(BC_TYPE)}"
    assert (
        psi_next.shape == psi_current.shape == psi_previous.shape == Q.shape
    ), "psi_next, psi_current, psi_previous, Q must have the same shape"

    if bctype in ["dirichlet", "neumann"] and bc_right_left_value is None:
        raise ValueError(
            "bc_right_left_value = (alpha, beta) must be provided for 'dirichlet','neumann' BC."
        )

    if bctype == "dirichlet":
        psi_next[0] = bc_right_left_value[0]
        psi_next[-1] = bc_right_left_value[1]

    elif bctype == "periodic":
        psi_next = 2.0 * psi_current - psi_previous
        psi_next += (c * dt / dx) ** 2 * (
            np.roll(psi_current, 1) - 2.0 * psi_current + np.roll(psi_current, -1)
        )
        psi_next += (c * dt) ** 2 * Q

    elif bctype == "transparent":
        tmp_coef = 1.0 / (dx + 3.0 * dt * c)
        psi_next[0] = tmp_coef * (
            4.0 * c * dt * psi_next[1] - c * dt * psi_next[2] + dx * psi_previous[0]
        )
        psi_next[-1] = tmp_coef * (
            4.0 * c * dt * psi_next[-2] - c * dt * psi_next[-3] + dx * psi_previous[-1]
        )

    elif bctype == "neumann":
        psi_next[0] = (1.0 / 3.0) * (
            2.0 * dx * bc_right_left_value[0] + 4.0 * psi_next[1] - psi_next[2]
        )
        psi_next[-1] = (1.0 / 3.0) * (
            2.0 * dx * bc_right_left_value[1] + 4.0 * psi_next[-2] - psi_next[-3]
        )

    return psi_next


#############################################################
#                                                           #
#                            MAIN                           #
#                                                           #
#############################################################
def main_experience(
    period_to_emulate: float = 0.5,
    T_nb_point: int = 501,
    space_interval: tuple[float, float] = (-1, 4),
    X_nb_point: int = 501,
    c: float = 3,
    psi: np.ndarray | callable = None,
    dtpsi: np.ndarray | callable = lambda x: 0 * x,
    bool_plot: bool = True,
    bctype: str = "transparent",
    bc_right_left_value: tuple[float, float] = None,
    Q: np.ndarray = None,
) -> np.ndarray:
    """
    Exact solution of the wave equation.

    Parameters
    ----------
    period_to_emulate : float, optional
        Time interval to emulate, by default 0.5.
    T_nb_point : int, optional
        Number of time points, by default 501.
    space_interval : tuple[float, float], optional
        Space interval, by default (-1, 4).
    X_nb_point : int, optional
        Number of space points, by default 501.
    c : float, optional
        Wave speed, by default 3.
    psi : np.ndarray | callable, optional
        Initial condition, by default None.
    dtpsi : np.ndarray | callable, optional
        Initial condition of the derivative, by default lambda x: 0 * x.
    bool_plot : bool, optional
        Plot or not, by default True.
    bctype : str, optional
        Boundary condition type, by default "transparent".
    bc_right_left_value : tuple[float, float], optional
        Boundary condition value, by default None.
    Q : np.ndarray, optional
        Source term, by default None.

    Returns
    -------
    np.ndarray
        The solution of the wave equation.
    """
    assert bctype in BC_TYPE, f"bctype must be one of {' ' .join(BC_TYPE)}"

    print("Making the simulation of wave equation...")
    # Space mesh
    interval_inf, interval_sup = space_interval
    x = np.linspace(interval_inf, interval_sup, X_nb_point)
    dx = (interval_sup - interval_inf) / (X_nb_point - 1)

    # Time mesh
    t = np.linspace(0, period_to_emulate, T_nb_point)
    dt = period_to_emulate / (T_nb_point - 1)

    # Initialization
    if isinstance(Q, np.ndarray):
        pass
    elif callable(Q):
        Q = Q(x)
    else:
        Q = np.zeros(X_nb_point)

    if isinstance(psi, np.ndarray):
        pass
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

    if isinstance(dtpsi, np.ndarray):
        pass
    elif callable(dtpsi):
        dtpsi = dtpsi(x)
    else:
        dtpsi = np.zeros(X_nb_point)

    psis = np.zeros((T_nb_point, X_nb_point))
    psis[0] = psi
    psis[1] = psi + dtpsi * dt

    for t_n in tqdm(range(2, T_nb_point), desc="Progress", unit="step"):
        psis[t_n][1:-1] = 2 * psis[t_n - 1][1:-1] - psis[t_n - 2][1:-1]
        psis[t_n][1:-1] += (c * dt / dx) ** 2 * (
            np.roll(psis[t_n - 1], 1)[1:-1]
            - 2 * psis[t_n - 1][1:-1]
            + np.roll(psis[t_n - 1], -1)[1:-1]
        )
        psis[t_n][1:-1] += (c * dt) ** 2 * Q[1:-1]
        psis[t_n] = apply_boundary_condition(
            psis[t_n],
            psis[t_n - 1],
            psis[t_n - 2],
            c,
            dt,
            dx,
            Q,
            bctype,
            bc_right_left_value,
        )

    if bool_plot:
        # plotting
        if any(element != 0 for element in Q):
            fig, axs = plt.subplots(1, 3, figsize=(20, 8))
            axs[2].plot(x, Q)
            axs[2].set_title("Source term Q")
            axs[2].set_xlabel("Space")
            axs[2].set_ylabel("Q")
        else:
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        # plot level map
        psm = axs[0].pcolormesh(
            x,
            t,
            psis,
            cmap="turbo",
        )
        fig.colorbar(psm, ax=axs[0])
        axs[0].grid(False)
        axs[0].set_xlabel("Space")
        axs[0].set_ylabel("Time")
        axs[0].set_title(
            r"Heatmap of $\psi_{simul}$" + f" with {bctype} bc for wave equation"
        )

        # plot energy evolution
        def compute_L2_norm(psis):
            return np.array([np.linalg.norm(psi) for psi in psis])

        initial_energy = np.linalg.norm(psis[0])
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
        plt.show()

    return psis

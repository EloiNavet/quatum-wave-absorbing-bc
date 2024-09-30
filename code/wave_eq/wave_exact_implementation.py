import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

plt.rcParams.update({"font.family": "serif"})
plt.style.use("bmh")


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
    dtpsi: np.ndarray | callable = None,
    bool_plot: bool = True,
) -> np.ndarray:
    """
    Main function to compute the exact solution of the wave equation.

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
        Speed of the wave, by default 3.
    psi : np.ndarray | callable, optional
        Initial condition, by default None.
    dtpsi : np.ndarray | callable, optional
        Initial condition for the derivative, by default None.
    bool_plot : bool, optional
        Whether to plot or not, by default True.

    Returns
    -------
    np.ndarray
        The solution of the wave equation.
    """
    print("Computing the exact solution of the wave equation...")
    # Space mesh
    interval_inf, interval_sup = space_interval
    x = np.linspace(interval_inf, interval_sup, X_nb_point)

    # Time mesh
    period_to_emulate = period_to_emulate
    t = np.linspace(0, period_to_emulate, T_nb_point)

    psis = np.zeros((T_nb_point, X_nb_point), dtype=complex)
    for i, ti in tqdm(enumerate(t), total=len(t), desc="Progress", unit="step"):
        psi_plus = psi(x + c * ti)
        psi_minus = psi(x - c * ti)
        if dtpsi is None:
            psis[i] = (psi_plus + psi_minus) / 2
        else:
            integral = np.array(
                [
                    np.trapz(
                        dtpsi(np.linspace(xj - c * ti, xj + c * ti, X_nb_point)),
                        np.linspace(xj - c * ti, xj + c * ti, X_nb_point),
                    )
                    for xj in x
                ]
            )
            psis[i] = (psi_plus + psi_minus + integral / c) / 2

    if bool_plot:
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

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
        axs[0].set_title(r"Heatmap of $|\psi_\text{exact}|$ for wave equation.")

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

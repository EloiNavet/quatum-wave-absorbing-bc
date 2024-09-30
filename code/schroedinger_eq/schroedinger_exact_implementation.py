import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use("bmh")
plt.rcParams.update({"font.family": "serif"})

#############################################################
#                                                           #
#                         UTILITARIES                       #
#                                                           #
#############################################################


def sech_article(x: float) -> np.array:
    """
    Initial condition function from the article.

    Parameters
    ----------
    x : float
        Position.

    Returns
    -------
    np.array
        Value of the function at x.
    """
    psi_0_x = 2 / np.cosh(np.sqrt(2) * x) * np.exp(1j * 15 / 2 * x)
    return psi_0_x


def H(t: np.array, x: np.array) -> np.array:
    """
    Kernel function for the convolution.

    Parameters
    ----------
    t : np.array
        Time mesh.
    x : np.array
        Space mesh.

    Returns
    -------
    np.array
        Value of the kernel at (t, x).
    """
    res = (
        (1 / (2 * np.sqrt(t * np.pi)))
        * np.exp(-1j * np.pi / 4)
        * np.exp(1j * (x**2) / (4 * t))
    )
    return res


#############################################################
#                                                           #
#                            MAIN                           #
#                                                           #
#############################################################
def main_experience(
    period_to_emulate: float = 2.0,
    T_nb_point: int = 501,
    space_interval: tuple[float, float] = (-10, 10),
    X_nb_point: int = 501,
    psi: callable[[float], np.ndarray | callable] = sech_article,
    convol_interval: tuple[float, float] = (-20, 20),
    convol_nb_point: int = 10000,
    bool_plot: bool = True,
) -> None | np.ndarray:
    """
    Compute exact solution of Schrodinger equation.

    Parameters
    ----------
    period_to_emulate : float, optional
        Time of the simulation, by default 2.0.
    T_nb_point : int, optional
        Number of points in time, by default 501.
    space_interval : tuple[float, float], optional
        Interval of space, by default (-10, 10).
    X_nb_point : int, optional
        Number of points in space, by default 501.
    psi : callable[[float], np.ndarray | callable], optional
        Initial condition, by default sech_article.
    convol_interval : tuple[float, float], optional
        Interval for the convolution, by default (-20, 20).
    convol_nb_point : int, optional
        Number of points in the convolution mesh, by default 10000.
    bool_plot : bool, optional
        Whether to plot the solution, by default True.

    Returns
    -------
    None | np.ndarray
        Solution of the Schrodinger equation.
    """
    print("Computing the exact solution of the Schrodinger equation...")

    # Space mesh
    x = np.linspace(space_interval[0], space_interval[1], X_nb_point)

    # Time mesh
    t = np.linspace(0, period_to_emulate, T_nb_point)

    # Compute exact solution by convolution
    z_convol = np.linspace(convol_interval[0], convol_interval[1], convol_nb_point)

    delta_z = z_convol[1] - z_convol[0]

    H_t = H(t[1:, np.newaxis], z_convol).astype(complex)

    psi_0_convol = psi(x[:, np.newaxis] - z_convol).astype(complex)

    res_exact = np.zeros((T_nb_point, X_nb_point), dtype=complex)
    res_exact[0] = psi(x)
    start_time = time.time()
    res_exact[1:, :] = delta_z * (H_t @ psi_0_convol.T)
    print(f"\tExact solution computed in {time.time() - start_time:.2f} s")

    # Visualization
    if bool_plot:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        psm = axs[0].pcolormesh(
            x,
            t,
            np.abs(res_exact),
            cmap="turbo",
        )
        fig.colorbar(psm, ax=axs[0])
        axs[0].grid(False)
        axs[0].set_xlabel("Space")
        axs[0].set_ylabel("Time")
        axs[0].set_title(r"Heatmap of $|\psi_\text{exact}|$ for Schrodinger equation.")

        def compute_L2_norm(psis):
            return np.array([np.linalg.norm(psi) for psi in psis])

        initial_energy = np.linalg.norm(res_exact[0])
        L2_energy = compute_L2_norm(res_exact)
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

    return res_exact

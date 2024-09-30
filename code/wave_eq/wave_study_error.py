import numpy as np
import matplotlib.pyplot as plt

from wave_eq.wave_simul_implementation import main_experience as wave_experience_simul
from wave_eq.wave_exact_implementation import main_experience as wave_experience_exact

plt.rcParams.update({"font.family": "serif"})
plt.style.use("bmh")


#############################################################
#                                                           #
#                            MAIN                           #
#                                                           #
#############################################################
def main_experience_error(
    period_to_emulate: float = 0.5,
    T_nb_point: int = 501,
    space_interval: tuple[float, float] = (-1, 4),
    X_nb_point: int = 501,
    c: float = 3,
    psi: np.ndarray | callable = None,
    dtpsi: np.ndarray | callable = None,
    bctype: str = "transparent",
    bc_right_left_value: tuple[float, float] = None,
) -> None:
    """
    Main function to compute the error of the wave equation.

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
    bctype : str, optional
        Boundary condition type, by default "transparent".
    bc_right_left_value : tuple[float, float], optional
        Boundary condition value, by default None.
    """
    res_simul = wave_experience_simul(
        period_to_emulate=period_to_emulate,
        T_nb_point=T_nb_point,
        space_interval=space_interval,
        X_nb_point=X_nb_point,
        c=c,
        psi=psi,
        dtpsi=dtpsi,
        bctype=bctype,
        bc_right_left_value=bc_right_left_value,
        bool_plot=True,
    )

    res_exact = wave_experience_exact(
        period_to_emulate=period_to_emulate,
        T_nb_point=T_nb_point,
        space_interval=space_interval,
        X_nb_point=X_nb_point,
        c=c,
        psi=psi,
        dtpsi=dtpsi,
        bool_plot=True,
    )

    # Space mesh
    interval_inf, interval_sup = space_interval
    x = np.linspace(interval_inf, interval_sup, X_nb_point)

    # Time mesh
    period_to_emulate = period_to_emulate
    t = np.linspace(0, period_to_emulate, T_nb_point)

    # Compute error and energy
    def compute_inf_norm(psis):
        return [np.max(np.abs(psi)) for psi in psis]

    # Plotting error
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    def relative_max_error_eval(res_exact, res_simul):
        # implementation of e_r
        abs_error = np.abs(res_exact - res_simul)
        max_abs_error = compute_inf_norm(abs_error)
        max_abs = np.max(compute_inf_norm(np.abs(res_exact)))
        max_abs_rel_error = max_abs_error / max_abs
        return max_abs_rel_error, abs_error

    max_abs_rel_error, abs_error = relative_max_error_eval(res_exact, res_simul)
    ind_bound_max_abs_rel_error = np.argmax(max_abs_rel_error)

    psm = axs[0].pcolormesh(
        x,
        t,
        abs_error,
        cmap="turbo",
    )
    fig.colorbar(psm, ax=axs[0])
    axs[0].grid(False)
    axs[0].set_xlabel("Space")
    axs[0].set_ylabel("Time")
    axs[0].set_title(
        r"Level map of error $\Delta (x,t) := |\psi_{exact}(x,t) - \psi_{simul}(x,t)|$"
    )

    (plot1,) = axs[1].plot(max_abs_rel_error, t, color="red", label=r"$e_r(t)$")
    (plot2,) = axs[1].plot(
        [max_abs_rel_error[ind_bound_max_abs_rel_error]]
        * len(t[:ind_bound_max_abs_rel_error]),
        t[:ind_bound_max_abs_rel_error],
        color="red",
        label=r"$\max_t (e_r(t))$",
        linestyle="dotted",
    )
    axs[1].plot(
        np.linspace(-0.001, max_abs_rel_error[ind_bound_max_abs_rel_error], 2),
        [t[ind_bound_max_abs_rel_error]] * 2,
        color="red",
        linestyle="dotted",
    )
    axs[1].set_ylim([0, period_to_emulate])
    axs[1].set_xlim([-0.001, 1.12 * np.max(max_abs_rel_error)])
    axs[1].set_xlabel(r"$e_r(t)$")

    plots = [plot1, plot2]
    labels = [line.get_label() for line in plots]
    axs[1].legend(plots, labels, loc="upper right")
    axs[1].set_title(
        r"Evolution of $e_r(t)$. "
        + f"Maximum reached: {max_abs_rel_error[ind_bound_max_abs_rel_error]:.5g}"
    )

    fig.suptitle(r"Evolution of the error for wave equation", fontsize=16)
    print("Error study summary:")
    print(
        "\t"
        + r"Simulation parameter: Delta_t="
        + f"{(period_to_emulate/T_nb_point)}\t"
        + "Delta_x="
        + f"{((space_interval[1] - space_interval[0])/X_nb_point)}\t"
    )
    print(
        f"\tMaximum error: {np.max(abs_error)}\t Maximum relative error: {max_abs_rel_error[ind_bound_max_abs_rel_error]}"
    )
    fig.tight_layout()
    plt.show()

    return res_exact, res_simul, abs_error


def isolate_reflections(
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
) -> np.ndarray:
    """
    Isolate reflections for the wave equation.

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
        Initial condition for the derivative, by default lambda x: 0 * x.
    bool_plot : bool, optional
        Whether to plot or not, by default True.
    bctype : str, optional
        Boundary condition type, by default "transparent".
    bc_right_left_value : tuple[float, float], optional
        Boundary condition value, by default None.

    Returns
    -------
    np.ndarray
        The isolated reflections for the wave equation.
    """
    interval_inf, interval_sup = space_interval
    x = np.linspace(interval_inf, interval_sup, X_nb_point)
    t = np.linspace(0, period_to_emulate, T_nb_point)

    res_basic_simul = wave_experience_simul(
        period_to_emulate=period_to_emulate,
        T_nb_point=T_nb_point,
        space_interval=space_interval,
        X_nb_point=X_nb_point,
        c=c,
        psi=psi,
        dtpsi=dtpsi,
        bctype=bctype,
        bc_right_left_value=bc_right_left_value,
        bool_plot=False,
    )

    # Perform same simulation BUT on larger domain to avoid reflections
    size_of_domain = interval_sup - interval_inf
    enlarged_space_interval = (
        interval_inf - size_of_domain,
        interval_sup + size_of_domain,
    )
    enlarged_X_nb_point = 3 * X_nb_point - 2

    res_enlarged_simul = wave_experience_simul(
        period_to_emulate=period_to_emulate,
        T_nb_point=T_nb_point,
        space_interval=enlarged_space_interval,
        X_nb_point=enlarged_X_nb_point,
        c=c,
        psi=psi,
        dtpsi=dtpsi,
        bctype=bctype,
        bc_right_left_value=bc_right_left_value,
        bool_plot=False,
    )

    # Truncate to get back to original domain size and isolate reflections
    res_enlarged_truncated = res_enlarged_simul[
        :, enlarged_X_nb_point // 3 : 1 + 2 * enlarged_X_nb_point // 3
    ]
    res_isolated_reflections = res_basic_simul - res_enlarged_truncated

    if bool_plot:
        # plotting
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        # plot level map
        psm = axs[0].pcolormesh(
            x,
            t,
            np.abs(res_isolated_reflections),
            cmap="turbo",
        )
        fig.colorbar(psm, ax=axs[0])
        axs[0].grid(False)
        axs[0].set_xlabel("Space")
        axs[0].set_ylabel("Time")
        axs[0].set_title(
            r"Heatmap of $||\psi-\phi|_{\Omega}||$"
            + f" with {bctype} bc for wave equation"
        )

        # plot energy evolution
        def compute_inf_norm(psis):
            return np.array([np.max(np.abs(psi)) for psi in psis])

        relative_error = compute_inf_norm(np.abs(res_isolated_reflections)) / np.max(
            compute_inf_norm(np.abs(res_basic_simul))
        )
        ind_bound_max_abs_rel_error = np.argmax(relative_error)

        (plot1,) = axs[1].plot(relative_error, t, color="red", label=r"$e_{r,i}(t)$")
        (plot2,) = axs[1].plot(
            [relative_error[ind_bound_max_abs_rel_error]]
            * len(t[:ind_bound_max_abs_rel_error]),
            t[:ind_bound_max_abs_rel_error],
            color="red",
            label=r"$\max_t (e_{r,i}(t))$",
            linestyle="dotted",
        )
        axs[1].plot(
            np.linspace(-0.001, relative_error[ind_bound_max_abs_rel_error], 2),
            [t[ind_bound_max_abs_rel_error]] * 2,
            color="red",
            linestyle="dotted",
        )
        axs[1].set_ylim([0, period_to_emulate])
        axs[1].set_xlim([-0.001, 1.12 * np.max(relative_error)])
        axs[1].set_xlabel(r"$e_{r,i}(t)$")

        plots = [plot1, plot2]
        labels = [line.get_label() for line in plots]
        axs[1].legend(plots, labels, loc="upper right")
        axs[1].set_title(
            r"Evolution of $e_{r,i}(t)$. "
            + f"Maximum reached: {relative_error[ind_bound_max_abs_rel_error]:.5g}"
        )

        fig.suptitle(r"Isolating reflections for wave equation", fontsize=16)
        print("Error study summary:")
        print(
            "\t"
            + r"Simulation parameter: Delta_t="
            + f"{(period_to_emulate/T_nb_point)}\t"
            + "Delta_x="
            + f"{((space_interval[1] - space_interval[0])/X_nb_point)}\t"
        )
        print(
            f"\tMaximum error: {np.max(np.abs(res_isolated_reflections))}\t Maximum relative error: {relative_error[ind_bound_max_abs_rel_error]}"
        )
        fig.tight_layout()
        plt.show()

    return res_isolated_reflections

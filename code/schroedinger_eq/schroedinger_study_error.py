import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")
plt.rcParams.update({"font.family": "serif"})

from schroedinger_eq.schroedinger_simul_implementation import (
    main_experience as schroedinger_simul_experience,
)
from schroedinger_eq.schroedinger_exact_implementation import (
    main_experience as schroedinger_exact_experience,
)


#############################################################
#                                                           #
#                         UTILITARIES                       #
#                                                           #
#############################################################
def sech_article(x: np.ndarray) -> np.ndarray:
    """
    Initial condition for the Schrödinger equation in the article.

    Parameters
    ----------
    x : np.array
        Space mesh.

    Returns
    -------
    np.array
        The initial condition.
    """
    psi_0_x = 2 / np.cosh(np.sqrt(2) * x) * np.exp(1j * 15 / 2 * x)  # See the article
    return psi_0_x


def H(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Exact solution for the Schrödinger equation.

    Parameters
    ----------
    t : np.array
        Time mesh.
    x : np.array
        Space mesh.

    Returns
    -------
    np.array
        The exact solution.
    """
    res = (
        (1 / (2 * np.sqrt(t * np.pi)))
        * np.exp(1j * np.pi / 4)
        * np.exp(1j * (x**2) / (4 * t))
    )  # to verif
    return res


#############################################################
#                                                           #
#                            MAIN                           #
#                                                           #
#############################################################
def study_of_error(
    period_to_emulate: float = 2,
    T_nb_point: int = 501,
    space_interval: tuple[float, float] = (-10, 10),
    X_nb_point: int = 501,
    psi: np.ndarray | callable = sech_article,
    bctype: str = "transparent_basic",
    bc_right_left_value: tuple[float, float] | None = [0, 0],
    activate_PML: bool = False,
    delta: tuple[float, float] = (5, 5),
    sigma_0: float = 0.05,
    V_independent: np.ndarray | None = None,
    V_non_linear_part: np.ndarray | None = None,
    convol_interval: tuple[float, float] = (-20, 20),
    convol_nb_point: int = 10000,
) -> np.ndarray | None:
    """
    Make the error computation for Schroedinguer equation in free space psi_0:x mapsto 2 / np.cosh(np.sqrt(2) * x) * np.exp(1j * 15 / 2 * x)

    Parameters
    ----------
    period_to_emulate : float, optional
        Time simulation duration, by default 2.
    T_nb_point : int, optional
        Number of points for time discretization, by default 501.
    space_interval : tuple[float, float], optional
        Space interval, by default (-10, 10).
    X_nb_point : int, optional
        Number of points for space discretization, by default 501.
    psi : np.ndarray | callable, optional
        Initial condition, must be a function, by default sech_article.
    bctype : str, optional
        Type of boundary condition, by default "transparent_basic".
    bc_right_left_value : tuple[float, float] | None, optional
        must be provided for 'dirichlet','neumann' BC, by default [0, 0].
    activate_PML : bool, optional
        specifiate PML or not, by default False.
    delta : tuple(float, float), optional
        size of the PML (included in space_interval), by default (5, 5).
    sigma_0 : float, optional
        parameter of the PML, by default 0.05.
    V_independent : Union[np.array, None], optional
        Potential, independant of psi, by default None.
    V_non_linear_part : Union[np.array, None], optional
        Potential, part which depends on psi, by default None.
    convol_interval : tuple[float, float], optional
        Integral convolution boundary, by default (-20, 20).
    convol_nb_point : int, optional
        Number of points for convolution product, by default 10000.

    Returns
    -------
    Union[None, np.array]
        the result of the computation if bool_return_result is True.
    """

    res_simul = schroedinger_simul_experience(
        period_to_emulate=period_to_emulate,
        T_nb_point=T_nb_point,
        space_interval=space_interval,
        X_nb_point=X_nb_point,
        psi=psi,
        bctype=bctype,
        bc_right_left_value=bc_right_left_value,
        activate_PML=activate_PML,
        delta=delta,
        sigma_0=sigma_0,
        V_independent=V_independent,
        V_non_linear_part=V_non_linear_part,
        bool_plot=True,
    )

    res_exact = schroedinger_exact_experience(
        period_to_emulate=period_to_emulate,
        T_nb_point=T_nb_point,
        space_interval=space_interval,
        X_nb_point=X_nb_point,
        psi=psi,
        convol_interval=convol_interval,
        convol_nb_point=convol_nb_point,
        bool_plot=False,
    )

    # Space mesh
    x = np.linspace(space_interval[0], space_interval[1], X_nb_point)
    dx = (space_interval[1] - space_interval[0]) / (X_nb_point - 1)

    if activate_PML:
        inner_index_left = int(delta[0] / dx)
        inner_index_right = int(delta[1] / dx)
        res_simul = res_simul[:, inner_index_left:-inner_index_right]
        res_exact = res_exact[:, inner_index_left:-inner_index_right]
        x = x[inner_index_left:-inner_index_right]

    print("Start computation of error")

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

    fig.suptitle(r"Evolution of the error for Schrödinger equation", fontsize=16)
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
    space_interval: tuple[float, float] = (-10, 10),
    X_nb_point: int = 501,
    psi: np.ndarray | callable = None,
    bctype: str = "transparent",
    bc_right_left_value: tuple[float, float] = [0, 0],
    activate_PML: bool = False,
    delta: tuple[float, float] = (5.0, 5.0),
    sigma_0: float = 0.05,
    V_independent: np.ndarray = None,
    V_non_linear_part: np.ndarray | callable = None,
    bool_plot: bool = True,
) -> np.ndarray:
    """
    Isolate reflections for Schrödinger equation.

    Parameters
    ----------
    period_to_emulate : float, optional
        Time simulation duration, by default 0.5.
    T_nb_point : int, optional
        Number of points for time discretization, by default 501.
    space_interval : tuple[float, float], optional
        Space interval, by default (-10, 10).
    X_nb_point : int, optional
        Number of points for space discretization, by default 501.
    psi : np.ndarray | callable, optional
        Initial condition, must be a function, by default None.
    bctype : str, optional
        Type of boundary condition, by default "transparent".
    bc_right_left_value : tuple[float, float], optional
        must be provided for 'dirichlet','neumann' BC, by default [0, 0].
    activate_PML : bool, optional
        specifiate PML or not, by default False.
    delta : tuple(float, float), optional
        size of the PML (included in space_interval), by default (5.0, 5.0).
    sigma_0 : float, optional
        parameter of the PML, by default 0.05.
    V_independent : np.array, optional
        Potential, independant of psi, by default None.
    V_non_linear_part : np.array | callable, optional
        Potential, part which depends on psi, by default None.
    bool_plot : bool, optional
        Plot or not the result, by default True.

    Returns
    -------
    np.array
        the result of the computation.
    """
    interval_inf, interval_sup = space_interval
    x = np.linspace(interval_inf, interval_sup, X_nb_point)
    dx = (space_interval[1] - space_interval[0]) / (X_nb_point - 1)
    t = np.linspace(0, period_to_emulate, T_nb_point)

    res_basic_simul = schroedinger_simul_experience(
        period_to_emulate=period_to_emulate,
        T_nb_point=T_nb_point,
        space_interval=space_interval,
        X_nb_point=X_nb_point,
        psi=psi,
        bctype=bctype,
        bc_right_left_value=bc_right_left_value,
        activate_PML=activate_PML,
        delta=delta,
        sigma_0=sigma_0,
        V_independent=V_independent,
        V_non_linear_part=V_non_linear_part,
        bool_plot=False,
    )

    # Perform same simulation BUT on larger domain to avoid refelctions
    size_of_domain = interval_sup - interval_inf
    enlarged_space_interval = (
        interval_inf - size_of_domain,
        interval_sup + size_of_domain,
    )
    enlarged_X_nb_point = 3 * X_nb_point - 2

    res_enlarged_simul = schroedinger_simul_experience(
        period_to_emulate=period_to_emulate,
        T_nb_point=T_nb_point,
        space_interval=enlarged_space_interval,
        X_nb_point=enlarged_X_nb_point,
        psi=psi,
        bctype=bctype,
        bc_right_left_value=bc_right_left_value,
        activate_PML=activate_PML,
        delta=delta,
        sigma_0=sigma_0,
        V_independent=V_independent,
        V_non_linear_part=V_non_linear_part,
        bool_plot=False,
    )

    # Truncate to get back to original domain size and isolate reflections
    res_enlarged_truncated = res_enlarged_simul[
        :, enlarged_X_nb_point // 3 : 1 + 2 * enlarged_X_nb_point // 3
    ]
    res_isolated_reflections = res_basic_simul - res_enlarged_truncated

    if bool_plot:
        # plotting
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))

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
            + f" with {bctype} bc"
            + activate_PML * " and PML"
            + " for Schrödinger equation"
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
            np.linspace(-0.00001, relative_error[ind_bound_max_abs_rel_error], 2),
            [t[ind_bound_max_abs_rel_error]] * 2,
            color="red",
            linestyle="dotted",
        )
        axs[1].set_ylim([0, period_to_emulate])
        axs[1].set_xlim([-0.00001, 1.12 * np.max(relative_error)])
        axs[1].set_xlabel(r"$e_{r,i}(t)$")

        plots = [plot1, plot2]
        labels = [line.get_label() for line in plots]
        axs[1].legend(plots, labels, loc="upper right")
        axs[1].set_title(
            r"Evolution of $e_{r,i}(t)$. "
            + f"Maximum reached: {relative_error[ind_bound_max_abs_rel_error]:.5g}"
        )
        fig.suptitle(
            r"Isolating reflections for Schrödinger equation inner", fontsize=16
        )

        if activate_PML:  # make a second plot zoomed
            axs[0].axvline(x=interval_inf + delta[0], color="r")
            axs[0].axvline(x=interval_sup - delta[1], color="r")
            fig.tight_layout()
            plt.show()

            inner_index_left = int(delta[0] / dx)
            inner_index_right = int(delta[1] / dx)

            res_enlarged_truncated = res_enlarged_truncated[
                :, inner_index_left:-inner_index_right
            ]
            res_basic_simul = res_basic_simul[:, inner_index_left:-inner_index_right]
            x = x[inner_index_left:-inner_index_right]

            res_isolated_reflections = res_basic_simul - res_enlarged_truncated

            # plotting
            fig, axs = plt.subplots(1, 2, figsize=(18, 6))

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
                r"Zoomed heatmap of $||\psi-\phi|_{\Omega}||$"
                + f" with {bctype} bc and PML for Schröedinger equation"
            )

            # plot energy evolution
            def compute_inf_norm(psis):
                return np.array([np.max(np.abs(psi)) for psi in psis])

            relative_error = compute_inf_norm(
                np.abs(res_isolated_reflections)
            ) / np.max(compute_inf_norm(np.abs(res_basic_simul)))
            ind_bound_max_abs_rel_error = np.argmax(relative_error)

            (plot1,) = axs[1].plot(
                relative_error, t, color="red", label=r"$e_{r,i}(t)$"
            )
            (plot2,) = axs[1].plot(
                [relative_error[ind_bound_max_abs_rel_error]]
                * len(t[:ind_bound_max_abs_rel_error]),
                t[:ind_bound_max_abs_rel_error],
                color="red",
                label=r"$\max_t (e_{r,i}(t))$",
                linestyle="dotted",
            )
            axs[1].plot(
                np.linspace(-0.00001, relative_error[ind_bound_max_abs_rel_error], 2),
                [t[ind_bound_max_abs_rel_error]] * 2,
                color="red",
                linestyle="dotted",
            )
            axs[1].set_ylim([0, period_to_emulate])
            axs[1].set_xlim([-0.00001, 1.12 * np.max(relative_error)])
            axs[1].set_xlabel(r"$e_{r,i}(t)$")

            plots = [plot1, plot2]
            labels = [line.get_label() for line in plots]
            axs[1].legend(plots, labels, loc="upper right")
            axs[1].set_title(
                r"Evolution of $e_{r,i}(t)$. "
                + f"Maximum reached: {relative_error[ind_bound_max_abs_rel_error]:.5g}"
            )
            fig.suptitle(
                r"Isolating reflections for Schrödinger equation inner", fontsize=16
            )

        fig.tight_layout()
        plt.show()
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

    return res_isolated_reflections

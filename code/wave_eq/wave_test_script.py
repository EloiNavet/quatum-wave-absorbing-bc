import numpy as np
from wave_study_error import main_experience_error

# Time mesh
period_to_emulate = 1
T_nb_point = 1000

# Space mesh
space_interval = (-1, 2)
X_nb_point = 500

# Celerity
c = 6

# Period to emulate
T = 1


# Initial condition (a default function is implemented)
def double_gaussian(x):
    psi_0_x = np.exp(
        -(
            (x - (space_interval[0] + 0.2 * (space_interval[1] - space_interval[0])))
            ** 2
        )
        / (0.1**2)
    )
    psi_0_x += np.exp(
        -(
            (x - (space_interval[0] + 0.6 * (space_interval[1] - space_interval[0])))
            ** 2
        )
        / (0.1**2)
    )
    return psi_0_x


psi_0 = double_gaussian

interval_inf = space_interval[0]
interval_sup = space_interval[1]
x = np.linspace(interval_inf, interval_sup, X_nb_point)


def double_gaussian_deriv(x):
    psi_0_x = (
        -2
        * (x - (space_interval[0] + 0.2 * (space_interval[1] - space_interval[0])))
        * np.exp(
            -(
                (
                    x
                    - (
                        space_interval[0]
                        + 0.2 * (space_interval[1] - space_interval[0])
                    )
                )
                ** 2
            )
            / (0.1**2)
        )
        / (0.1**2)
    )
    psi_0_x += (
        -2
        * (x - (space_interval[0] + 0.6 * (space_interval[1] - space_interval[0])))
        * np.exp(
            -(
                (
                    x
                    - (
                        space_interval[0]
                        + 0.6 * (space_interval[1] - space_interval[0])
                    )
                )
                ** 2
            )
            / (0.1**2)
        )
        / (0.1**2)
    )
    return -c * psi_0_x


dtpsi_0 = double_gaussian_deriv
x = np.linspace(space_interval[0], space_interval[1], X_nb_point)

main_experience_error(
    period_to_emulate = T,
    T_nb_point = T_nb_point,
    space_interval = space_interval,
    X_nb_point = X_nb_point,
    c = c,
    psi_0 = psi_0,
    dtpsi_0 = dtpsi_0,
    bctype = "transparent",
)

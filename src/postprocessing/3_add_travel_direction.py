"""
As a third additional post-processing step, it is useful to include the
direction of spacecraft travel as part of the crossings list. We can estimate
this by comparing the spacecraft's velocity vector (obtained via SPICE), with
an estimate of the boundary's normal vector at the position of the crossing -
determined as the opposite of the vector to the closest point on the average
boundary curve.
"""

import datetime as dt
from typing import Literal, Tuple

import hermpy.trajectory
import hermpy.utils
import numpy as np
import pandas as pd
import scipy
import spiceypy as spice
from numpy.typing import NDArray


def main():

    hermpy.utils.User.METAKERNEL = "./SPICE/messenger/metakernel_messenger.txt"

    # Load the crossings
    crossings = pd.read_csv(
        "./data/postprocessing/2_include_hidden_crossings.csv",
        parse_dates=["Time"],
    )

    # Get the position of all crossings
    positions, velocities = get_positions_and_velocities(crossings)

    crossing_labels = crossings["Label"]

    direction_labels = []
    position_vectors = []
    normal_vectors = []
    velocity_vectors = []

    for p, v, label in zip(positions, velocities, crossing_labels):

        if "BS" in label:
            n = get_boundary_normal(p, boundary="Bow Shock")
        elif "MP" in label:
            n = get_boundary_normal(p, boundary="Magnetopause")

        # If the dot product of the velocity and normal is positive, it is an
        # outbound crossing, if it is negative, it is an inbound crossing.
        is_outbound = np.dot(v, n) > 0

        if is_outbound:
            direction_labels.append("Outwards")

        else:
            direction_labels.append("Inwards")

        position_vectors.append(p)
        normal_vectors.append(n)
        velocity_vectors.append(v)

    crossings["Trajectory Direction"] = direction_labels
    # crossings["Positions"] = position_vectors
    # crossings["Normal Vector"] = normal_vectors
    # crossings["Velocity Vector"] = velocity_vectors

    crossings.to_csv(
        "./data/postprocessing/hollman_2026_crossing_list.csv", index=False
    )


def get_positions_and_velocities(crossings) -> Tuple[NDArray, NDArray]:

    # Metakernel was created with scripts/make_messenger_metakernel
    with spice.KernelPool(hermpy.utils.User.METAKERNEL):

        crossing_positions = hermpy.trajectory.Get_Position(
            "MESSENGER", crossings["Time"], frame="MSM"
        )

        # Calculate a time 1 second prior and subtract to get velocity. We
        # don't have to worry about averages here as MESSENGER's velocity
        # should be smooth and continous.
        previous_positions = hermpy.trajectory.Get_Position(
            "MESSENGER", crossings["Time"] - dt.timedelta(seconds=1), frame="MSM"
        )

    # Divide by radius
    crossing_positions /= hermpy.utils.Constants.MERCURY_RADIUS_KM
    previous_positions /= hermpy.utils.Constants.MERCURY_RADIUS_KM

    # Convert to cylindricals
    crossing_positions = np.column_stack(
        (
            crossing_positions[:, 0],
            np.sqrt(crossing_positions[:, 1] ** 2 + crossing_positions[:, 2] ** 2),
        )
    )
    previous_positions = np.column_stack(
        (
            previous_positions[:, 0],
            np.sqrt(previous_positions[:, 1] ** 2 + previous_positions[:, 2] ** 2),
        )
    )

    velocities = crossing_positions - previous_positions

    # Normalise velocities to just get direction
    velocities = np.array([v / np.linalg.norm(v) for v in velocities])

    return crossing_positions, velocities


def get_boundary_normal(
    position: Tuple[float, float], boundary: Literal["Bow Shock", "Magnetopause"]
) -> Tuple[float, float]:

    match boundary:
        case "Bow Shock":
            initial_x = 0.5
            psi = 1.04
            p = 2.75

            L = psi * p

            phi = np.linspace(0, 2 * np.pi, 10000)
            rho = L / (1 + psi * np.cos(phi))

            # Cylindrical coordinates (X, R)
            x_coords = initial_x + rho * np.cos(phi)
            r_coords = rho * np.sin(phi)

            boundary_positions = np.array([x_coords, r_coords]).T

        case "Magnetopause":
            sub_solar_point = 1.45
            alpha = 0.5

            phi = np.linspace(0, 2 * np.pi, 10000)
            rho = sub_solar_point * (2 / (1 + np.cos(phi))) ** alpha

            # Cylindrical coordinates (X, R)
            x_coords = rho * np.cos(phi)
            r_coords = rho * np.sin(phi)

            boundary_positions = np.array([x_coords, r_coords]).T

        case _:
            raise ValueError(
                f"Invalid function choice: {boundary}. Options are 'Bow Shock', 'Magnetopause'."
            )

    # We need to determine which point on the boundary curve is closest to the spacecraft
    # This method, utilising a k-d tree is computationally faster than iterrating through
    # the points and determining the distance.
    # O(logN) vs O(N)
    kd_tree = scipy.spatial.KDTree(boundary_positions)
    _, closest_index = kd_tree.query(position)
    closest_position = boundary_positions[closest_index]

    # We can numerically find the tangent at each point on the curve using the
    # gradient.
    dx_dphi = np.gradient(x_coords, phi)
    dr_dphi = np.gradient(r_coords, phi)

    tangent = np.array([dx_dphi[closest_index], dr_dphi[closest_index]])
    tangent = tangent / np.sqrt(np.sum(tangent**2))

    # Rotate tangent by 90 degrees to get a normal candidate.
    normal_vector = np.array([-tangent[1], tangent[0]])

    # Ensure the normal points outwards by determining the dot product between
    # the candidate normal vector and the vector to the closest position on the
    # curve.
    if np.dot(normal_vector, closest_position) < 0:
        normal_vector = -normal_vector

    return normal_vector


if __name__ == "__main__":
    main()

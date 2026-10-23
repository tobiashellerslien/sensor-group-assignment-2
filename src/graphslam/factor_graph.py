"""Factor graph construction, measurement prediction and covariance assembly.
Functions marked ``TODO`` are Task 1 (c) to (g); the rest is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np

from graphslam.config import SlamConfig

# ---------------------------------------------------------------------------
# Noise models  (given)
# ---------------------------------------------------------------------------


def prior_noise_model(config: SlamConfig) -> gtsam.noiseModel.Base:
    """Noise model for the prior on X(0), ordered [x, y, yaw]."""
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.array(
            [
                config.noise.sigma_init_pose_x,
                config.noise.sigma_init_pose_y,
                config.noise.sigma_init_pose_yaw_rad,
            ]
        )
    )


def bearing_range_noise_model(config: SlamConfig) -> gtsam.noiseModel.Base:
    """Noise model for landmark factors, ordered [bearing, range] to match
    ``BearingRangeFactor2D``.
    """
    base = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([config.noise.sigma_bearing_rad, config.noise.sigma_range])
    )

    if config.backend.robust_kernel == "huber":
        return gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(config.backend.huber_k),
            base,
        )

    return base


# ---------------------------------------------------------------------------
# Task 1 (c): growing the graph with odometry
# ---------------------------------------------------------------------------


def predict_pose(previous_pose: gtsam.Pose2, relative_pose: gtsam.Pose2) -> gtsam.Pose2:
    """Initial guess for x_k: the previous pose composed with the increment. Task 1 (c1).

    Parameters
    ----------
    previous_pose : gtsam.Pose2
        The current estimate of the previous pose, x_{k-1}.
    relative_pose : gtsam.Pose2
        The measured body-frame increment, u_k.

    Returns
    -------
    gtsam.Pose2
        The predicted pose x_k.
    """
    return previous_pose.compose(relative_pose)


def add_odometry_factor(
    graph: gtsam.NonlinearFactorGraph,
    key_previous: int,
    key_current: int,
    relative_pose: gtsam.Pose2,
    relative_pose_cov: np.ndarray,
) -> None:
    """Add the odometry factor between x_{k-1} and x_k to the graph. Task 1 (c2).

    Parameters
    ----------
    graph : gtsam.NonlinearFactorGraph
        Graph of new factors, to be handed to the solver.
    key_previous, key_current : int
        Keys of x_{k-1} and x_k, i.e. ``X(k-1)`` and ``X(k)``.
    relative_pose : gtsam.Pose2
        Measured (preintegrated) increment.
    relative_pose_cov : np.ndarray, shape=(3, 3)
        Covariance of that increment, in the tangent space of ``relative_pose``.
    """
    noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_pose_cov)
    factor = gtsam.BetweenFactorPose2(
        key_previous,
        key_current,
        relative_pose,
        noise_model,
    )
    graph.add(factor)


def add_landmark_factor(
    graph: gtsam.NonlinearFactorGraph,
    pose_key: int,
    landmark_key: int,
    measurement: np.ndarray,
    noise_model: gtsam.noiseModel.Base,
) -> None:
    """Add one landmark measurement factor to the graph. Task 1 (c3).

    Parameters
    ----------
    graph : gtsam.NonlinearFactorGraph
        Graph of new factors, to be handed to the solver.
    pose_key, landmark_key : int
        Keys of the pose the measurement was taken from and of the landmark it
        was associated to.
    measurement : np.ndarray, shape=(2,)
        The measurement, **ordered [range, bearing]**.
    noise_model : gtsam.noiseModel.Base
        From :func:`bearing_range_noise_model`, ordered [bearing, range].
    """

    range, bearing = measurement
    graph.add(gtsam.BearingRangeFactor2D(
        pose_key,
        landmark_key,
        gtsam.Rot2(bearing),
        range,
        noise_model
    ))


# ---------------------------------------------------------------------------
# Task 1 (d): the measurement model and its Jacobians
# ---------------------------------------------------------------------------


def predict_measurement(
    pose: gtsam.Pose2,
    landmark: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted range-bearing measurement of one landmark, with Jacobians. Task 1 (d).

    Parameters
    ----------
    pose : gtsam.Pose2
        Current pose estimate x_k.
    landmark : np.ndarray, shape=(2,)
        Landmark position m_j in the world frame.

    Returns
    -------
    z : np.ndarray, shape=(2,)
        Predicted measurement, **ordered [range, bearing]**.
    H_pose : np.ndarray, shape=(2, 3)
        d z / d x_k, in the tangent space at ``pose``.
    H_landmark : np.ndarray, shape=(2, 2)
        d z / d m_j.
    """
    # TODO(d): predict [range, bearing] and stack the two rows of each Jacobian.
    raise NotImplementedError("Task 1 (d)")


def inverse_measurement(pose: gtsam.Pose2, measurement: np.ndarray) -> np.ndarray:
    """World-frame landmark position from one range-bearing measurement. Task 1 (g).

    Parameters
    ----------
    pose : gtsam.Pose2
        Pose the measurement was taken from.
    measurement : np.ndarray, shape=(2,)
        The measurement, **ordered [range, bearing]**.

    Returns
    -------
    np.ndarray, shape=(2,)
        Landmark position in the world frame.
    """
    # TODO(g): rotate by the bearing, translate by the range, map to the world frame.
    raise NotImplementedError("Task 1 (g)")


# ---------------------------------------------------------------------------
# Task 1 (e): the joint covariance of the local map
# ---------------------------------------------------------------------------


def assemble_joint_covariance(
    joint_marginal: gtsam.JointMarginal,
    keys: list[int],
) -> np.ndarray:
    """Assemble the joint covariance P over ``keys`` from a GTSAM joint marginal. Task 1 (e).

    Parameters
    ----------
    joint_marginal : gtsam.JointMarginal
        The joint marginal over ``keys``, as returned by
        ``isam2.jointMarginalCovariance`` or ``gtsam.Marginals(...).jointMarginalCovariance``.
    keys : list[int]
        The variables, in the order the blocks of P must follow: the pose first,
        then the local landmarks, e.g. ``[X(k), L(3), L(7)]``.

    Returns
    -------
    np.ndarray, shape=(3 + 2n, 3 + 2n)
        P, whose block (i, j) is the covariance between ``keys[i]`` and ``keys[j]``.
    """
    return np.block([[joint_marginal.at(key_i, key_j) for key_j in keys] for key_i in keys])


# ---------------------------------------------------------------------------
# Task 1 (f): the innovation covariance
# ---------------------------------------------------------------------------


def innovation_covariance(
    jacobians_pose: list[np.ndarray],
    jacobians_landmark: list[np.ndarray],
    joint_covariance: np.ndarray,
    measurement_cov: np.ndarray,
) -> np.ndarray:
    """Innovation covariance ``S = H P H^T + R`` for the whole local map. Task 1 (f).

    Parameters
    ----------
    jacobians_pose : list of np.ndarray, each shape=(2, 3)
        ``H_pose`` from :func:`predict_measurement`, one per local landmark, in
        the same order as the landmark blocks of ``joint_covariance``.
    jacobians_landmark : list of np.ndarray, each shape=(2, 2)
        ``H_landmark`` from :func:`predict_measurement`, same order.
    joint_covariance : np.ndarray, shape=(3 + 2n, 3 + 2n)
        Joint marginal ``P`` over ``[x_k, m_1, ..., m_n]``, pose block first.
    measurement_cov : np.ndarray, shape=(2, 2)
        Single-measurement ``R``, ordered [range, bearing].

    Returns
    -------
    np.ndarray, shape=(2n, 2n)
        The innovation covariance ``S``.
    """
    # TODO(f): build the stacked H, build the block-diagonal R, and form S.
    raise NotImplementedError("Task 1 (f)")


# ---------------------------------------------------------------------------
# Local map extraction  (given -- builds on your predict_measurement)
# ---------------------------------------------------------------------------


@dataclass
class LocalMap:
    """The landmarks currently in view, and everything needed to gate them."""

    keys: list[int] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    predicted_measurements: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    jacobians_pose: list[np.ndarray] = field(default_factory=list)
    jacobians_landmark: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.keys)


def extract_local_map(
    pose: gtsam.Pose2,
    landmark_keys: list[int],
    landmark_positions: dict[int, np.ndarray],
    config: SlamConfig,
) -> LocalMap:
    """Select the landmarks that could plausibly have produced a measurement.

    Restricting association to a local window is what keeps the covariance
    recovery in Task 1 (e) affordable: its cost grows with the number of keys in
    the query, not with the size of the map.
    """
    local = LocalMap()
    positions: list[np.ndarray] = []
    predicted: list[np.ndarray] = []

    for key in landmark_keys:
        landmark = landmark_positions[key]
        z, H_pose, H_landmark = predict_measurement(pose, landmark)
        predicted_range, predicted_bearing = z

        in_range = predicted_range < config.sensor.range_local
        in_field_of_view = abs(predicted_bearing) < config.sensor.bearing_local_rad

        if in_range and in_field_of_view:
            local.keys.append(key)
            positions.append(np.asarray(landmark, dtype=float).reshape(2))
            predicted.append(z)
            local.jacobians_pose.append(H_pose)
            local.jacobians_landmark.append(H_landmark)

    local.positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    local.predicted_measurements = np.asarray(predicted, dtype=float).reshape(-1, 2)

    return local

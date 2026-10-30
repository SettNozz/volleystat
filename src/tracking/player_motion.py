"""Constant-velocity image-space Kalman filter for observed player centres.

Time is measured in seconds, including negative time in the offline reverse
pass. Uncertainty grows during gaps; predictions never become exported poses.
"""
from __future__ import annotations

import numpy as np


class PlayerMotionKalman:
    def __init__(self, position, height: float, time: float):
        self.height = max(float(height), 1.)
        self.time = float(time)
        self.mean = np.r_[np.asarray(position, dtype=float), np.zeros(2)]
        self.covariance = np.diag(np.square(
            [self.height*.08]*2 + [self.height*.8]*2))

    def predict(self, time: float):
        dt = float(time)-self.time
        transition = np.eye(4)
        transition[:2, 2:] = np.eye(2)*dt
        # Integrated continuous white acceleration. Unlike a per-step random
        # acceleration draw, this gives the same uncertainty for a one-second
        # gap whether prediction is requested once or at 30 intermediate frames.
        elapsed = abs(dt)
        noise = np.zeros((4, 4))
        noise[:2, :2] = np.eye(2)*elapsed**3/3.
        noise[:2, 2:] = np.eye(2)*dt*elapsed/2.
        noise[2:, :2] = noise[:2, 2:]
        noise[2:, 2:] = np.eye(2)*elapsed
        noise *= (self.height*.8)**2
        return (transition @ self.mean,
                transition @ self.covariance @ transition.T + noise)

    def innovation(self, position, height: float, time: float, occlusion: float):
        mean, covariance = self.predict(time)
        scale = (self.height+max(float(height), 1.))/2.
        variance = (scale*.06*(1.+4.*float(occlusion)))**2
        residual = np.asarray(position)-mean[:2]
        measurement = np.eye(2)*variance
        distance = float(residual @ np.linalg.solve(
            covariance[:2, :2]+measurement, residual))
        return distance, mean, covariance, measurement, residual

    def update(self, position, height: float, time: float, occlusion: float):
        _, mean, covariance, measurement, residual = self.innovation(
            position, height, time, occlusion)
        gain = np.linalg.solve(covariance[:2, :2]+measurement,
                               covariance[:2, :]).T
        self.mean = mean + gain @ residual
        # Joseph form keeps covariance positive under repeated updates.
        remaining = np.eye(4)
        remaining[:, :2] -= gain
        self.covariance = (remaining @ covariance @ remaining.T
                           + gain @ measurement @ gain.T)
        self.time = float(time)
        self.height = .9*self.height+.1*max(float(height), 1.)

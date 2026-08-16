import numpy as np


class PIController:
    """
    PI controller for adaptive spatial search.

    The controller converts the matching error into an adaptive
    search-window adjustment.

    error      = target confidence - current confidence
    integral   = accumulated error
    control    = Kp * error + Ki * integral
    """

    def __init__(
        self,
        kp=40.0,
        ki=5.0,
        integral_limit=20.0,
        min_window=50.0,
        max_window=500.0,
    ):
        self.kp = kp
        self.ki = ki

        self.integral_limit = integral_limit

        self.min_window = min_window
        self.max_window = max_window

        self.integral = 0.0

    def reset(self):
        """Reset accumulated controller state."""
        self.integral = 0.0

    def update(self, target_confidence, measured_confidence):
        """
        Update controller using the current matching confidence.

        Positive error means confidence is below the desired level,
        therefore the controller increases the search window.
        """

        error = target_confidence - measured_confidence

        self.integral += error

        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit,
        )

        control = (
            self.kp * error
            + self.ki * self.integral
        )

        return control

    def adapt_window(
        self,
        current_window,
        target_confidence,
        measured_confidence,
    ):
        """
        Adapt the spatial search window based on ZNCC confidence.
        """

        control = self.update(
            target_confidence,
            measured_confidence,
        )

        new_window = current_window + control

        new_window = np.clip(
            new_window,
            self.min_window,
            self.max_window,
        )

        return float(new_window)


class NavigationPIController:
    """
    2-axis PI controller for navigation position correction.

    Given a localization error (target - predicted) in x and y,
    computes a corrective command using proportional and integral terms:

        correction = Kp * error + Ki * integral(error)

    This controller is independent of the visual localization pipeline.
    It consumes the localization error as input and produces a navigation
    correction as output.
    """

    def __init__(self, Kp=0.5, Ki=0.1, dt=1.0,
                 integral_limit=50.0, output_limit=100.0):
        """
        Parameters
        ----------
        Kp : float
            Proportional gain.
        Ki : float
            Integral gain.
        dt : float
            Time step for integral accumulation.
        integral_limit : float
            Anti-windup clamp for accumulated integral error.
        output_limit : float
            Clamp for the final corrective output.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.Kp = Kp
        self.Ki = Ki
        self.dt = dt
        self.integral_limit = integral_limit
        self.output_limit = output_limit

        self.integral_x = 0.0
        self.integral_y = 0.0

    def reset(self):
        """Reset accumulated integral error."""
        self.integral_x = 0.0
        self.integral_y = 0.0

    @staticmethod
    def _clip(val, limit):
        return max(-limit, min(limit, val))

    def update(self, error_x, error_y):
        """
        Compute the corrective command for the current error.

        Parameters
        ----------
        error_x : float
            target_x - predicted_x
        error_y : float
            target_y - predicted_y

        Returns
        -------
        correction_x, correction_y : float, float
        """
        # Accumulate integral
        self.integral_x += error_x * self.dt
        self.integral_y += error_y * self.dt

        # Anti-windup
        self.integral_x = self._clip(self.integral_x, self.integral_limit)
        self.integral_y = self._clip(self.integral_y, self.integral_limit)

        # PI output
        cx = self.Kp * error_x + self.Ki * self.integral_x
        cy = self.Kp * error_y + self.Ki * self.integral_y

        # Output clamp
        cx = self._clip(cx, self.output_limit)
        cy = self._clip(cy, self.output_limit)

        return cx, cy
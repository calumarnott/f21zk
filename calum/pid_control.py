import numpy as np

#reusable PID class
class PIDController:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, u_min=None, u_max=None):
        self.Kp = Kp #proportional gain
        self.Ki = Ki #integral gain
        self.Kd = Kd #derivative gain
        self.u_min = u_min #saturation limits of actuator
        self.u_max = u_max

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self): #generic reset function so we don't carry over stale error into new sims etc
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        # Proportional
        P = self.Kp * error

        # Integral
        self.integral += error * dt
        I = self.Ki * self.integral

        # Derivative
        D = self.Kd * (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        # Compute PID output (before clamp)
        u_unclamped = P + I + D

        # Apply saturation --> our actuators have limits after all!
        u = u_unclamped
        if self.u_min is not None:
            u = max(self.u_min, u)
        if self.u_max is not None:
            u = min(self.u_max, u)

        # Anti-windup: only integrate when not saturated OR when error drives output back toward range
        if (self.u_min is None or u_unclamped > self.u_min or error < 0) and \
                (self.u_max is None or u_unclamped < self.u_max or error > 0):
            self.integral += error * dt

        return u

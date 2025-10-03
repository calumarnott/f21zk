import numpy as np

#reusable PID class
class PIDController:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, u_min=None, u_max=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.u_min = u_min
        self.u_max = u_max

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
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

        # Output
        u = P + I + D

        # Clamp
        if self.u_min is not None:
            u = max(self.u_min, u)
        if self.u_max is not None:
            u = min(self.u_max, u)

        return u

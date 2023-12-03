class PID:
    def __init__(self, kp, ki, kd):
        """
        Initialize the PID controller.

        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        """
        # initialize gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # initialize error
        self.error = 0
        self.last_error = 0

        # initialize integral
        self.integral = 0

    def compute(self, value, setpoint, dt):
        """
        Compute the control signal with anti-windup.
        :param value: is the actual value of the system
        :param setpoint: is the desired goal value
        :param dt: is the time difference between the current and last time
        """

        # calculate error
        self.error = setpoint - value

        # calculate integral with anti-windup
        self.integral = self.integral + self.error * dt
        self.integral = max(min(self.integral, 1), -1)

        # calculate derivative but avoid inf or nan
        derivative = ((self.error - self.last_error) / dt) if dt > 0 else 0
        derivative = max(min(derivative, 1), -1)

        # calculate control
        control = self.kp * self.error + self.ki * self.integral + self.kd * derivative

        # update last error
        self.last_error = self.error

        return control
    
    
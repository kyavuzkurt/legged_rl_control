import numpy as np
from typing import Tuple

class PIDController:
    """
    Discrete-time PID controller for joint position control with anti-windup
    and configurable output limits.
    
    Args:
        Kp (float): Proportional gain
        Ki (float): Integral gain
        Kd (float): Derivative gain
        dt (float): Time step in seconds
        output_limits (Tuple[float, float]): Minimum and maximum output values
        integral_clamp (Tuple[float, float]): Integral anti-windup clamp range
    """
    def __init__(self, 
                 Kp: float = 1.0,
                 Ki: float = 0.0,
                 Kd: float = 0.0,
                 dt: float = 0.005,
                 output_limits: Tuple[float, float] = (-np.inf, np.inf),
                 integral_clamp: Tuple[float, float] = (-np.inf, np.inf)):
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limits = output_limits
        self.integral_clamp = integral_clamp
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0

    def compute(self, 
                setpoint: float, 
                measurement: float,
                measurement_deriv: float = None) -> float:
        """
        Compute PID output given current setpoint and measurement.
        
        Args:
            setpoint: Desired target value
            measurement: Current measured value
            measurement_deriv: Optional pre-computed derivative measurement
            
        Returns:
            float: Control output
        """
        error = setpoint - measurement
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup clamping
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, *self.integral_clamp)
        I = self.Ki * self.integral
        
        # Derivative term (either use provided derivative or estimate)
        if measurement_deriv is not None:
            D = self.Kd * (-measurement_deriv)
        else:
            D = self.Kd * (error - self.prev_error) / self.dt

        # Compute raw output
        output = P + I + D
        
        # Apply output limits
        output_clipped = np.clip(output, *self.output_limits)
        
        # Update previous values
        self.prev_error = error
        self.prev_measurement = measurement
        
        return output_clipped

    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0

def create_pid_controllers_from_config(controller_config: dict, num_joints: int, dt: float):
    """
    Factory function to create multiple PID controllers from configuration.
    
    Args:
        controller_config: Dictionary with PID parameters
        num_joints: Number of joints to create controllers for
        dt: Control timestep in seconds
        
    Returns:
        List[PIDController]: List of initialized PID controllers
    """
    return [
        PIDController(
            Kp=controller_config["Kp"],
            Ki=controller_config["Ki"],
            Kd=controller_config["Kd"],
            dt=dt,
            output_limits=controller_config["output_limits"],
            integral_clamp=controller_config.get("integral_clamp", [-np.inf, np.inf])
        ) for _ in range(num_joints)
    ] 
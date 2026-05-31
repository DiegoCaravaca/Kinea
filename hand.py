from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory


class Hand:
    """Controls the servo that opens and closes the robotic hand."""

    def __init__(self, port, open_angle, close_angle):
        """
        Args:
            port: (int) GPIO pin driving the servo (BCM numbering).
            open_angle: (int) servo angle for the open position.
            close_angle: (int) servo angle for the closed position.
        """
        # Pigpio gives hardware-timed pulses, which reduces servo jitter
        self.movement = AngularServo(
            port, min_angle=0, max_angle=180,
            min_pulse_width=0.0005, max_pulse_width=0.0025,
            pin_factory=PiGPIOFactory(),
        )
        self.open_angle = open_angle
        self.close_angle = close_angle
        self.is_open = None

    def open(self):
        """Move the hand to the open position. Skips if already open."""
        if self.is_open is not True:
            self.movement.angle = self.open_angle
            self.is_open = True
            print("\n  >>> HAND OPENED <<<")

    def close(self):
        """Move the hand to the closed position. Skips if already closed."""
        if self.is_open is not False:
            self.movement.angle = self.close_angle
            self.is_open = False
            print("\n  >>> HAND CLOSED <<<")

    def get_hand_state(self):
        """Return the current hand state.

        Returns:
            is_open: (bool|None) True if open, False if closed, None if unset.
        """
        return self.is_open

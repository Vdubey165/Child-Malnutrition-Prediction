"""
GPIO risk-indicator service — drives the 3-LED (Low/Medium/High) physical
indicator on the Raspberry Pi field-deployment build.

Import of RPi.GPIO is guarded: that library only exists on actual Raspberry
Pi hardware. Everywhere else this code runs (your laptop, Cloud Run), the
import fails, ON_PI is set False, and show_risk() becomes a silent no-op —
so this file is always safe to import and call from prediction.py
regardless of environment, and the cloud deployment is completely
unaffected by hardware code it will never execute.
"""
import logging

logger = logging.getLogger(__name__)

PINS = {"Low": 17, "Medium": 27, "High": 22}  # BCM numbering — see wiring diagram

try:
    import RPi.GPIO as GPIO

    GPIO.setmode(GPIO.BCM)
    for _pin in PINS.values():
        GPIO.setup(_pin, GPIO.OUT)
        GPIO.output(_pin, GPIO.LOW)

    ON_PI = True
    logger.info("GPIO risk indicator initialized — running on Raspberry Pi.")
except (ImportError, RuntimeError):
    # ImportError: RPi.GPIO not installed (not on a Pi at all).
    # RuntimeError: RPi.GPIO installed but not actually running on Pi hardware.
    ON_PI = False
    logger.info("RPi.GPIO not available — risk indicator disabled (expected off-Pi).")


def show_risk(risk_level: str) -> None:
    """Light the LED matching risk_level ('Low' | 'Medium' | 'High').
    No-op if not running on the Pi."""
    if not ON_PI:
        return
    if risk_level not in PINS:
        logger.warning("show_risk() called with unrecognized level: %s", risk_level)
        return
    for level, pin in PINS.items():
        GPIO.output(pin, GPIO.HIGH if level == risk_level else GPIO.LOW)


def clear() -> None:
    """Turn off all LEDs. No-op if not running on the Pi."""
    if not ON_PI:
        return
    for pin in PINS.values():
        GPIO.output(pin, GPIO.LOW)

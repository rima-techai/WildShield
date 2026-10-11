from __future__ import annotations

import os
import sys
import time

MLX90640_ADDRESS = 0x33


def run() -> int:
    if os.environ.get("BLINKA_MCP2221") != "1":
        print("error: set BLINKA_MCP2221=1 before running this command", file=sys.stderr)
        return 2
    try:
        import board
        import adafruit_mlx90640
        i2c = board.I2C()
        while not i2c.try_lock():
            time.sleep(0.01)
        try:
            addresses = i2c.scan()
        finally:
            i2c.unlock()
        formatted = ", ".join(f"0x{address:02X}" for address in addresses) or "none"
        print(f"I2C addresses found: {formatted}")
        if MLX90640_ADDRESS not in addresses:
            print("error: MLX90640 was not found at expected address 0x33; check the STEMMA QT connections and reconnect USB power", file=sys.stderr)
            return 1
        sensor = adafruit_mlx90640.MLX90640(i2c)
        sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        frame = [0.0] * 768
        deadline = time.monotonic() + 10.0
        while True:
            try:
                sensor.getFrame(frame)
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for a complete thermal frame")
                time.sleep(0.1)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: thermal hardware check failed: {exc}", file=sys.stderr)
        return 1
    center = frame[(12 * 32) + 16]
    print("MLX90640 detected at 0x33")
    print(f"Frame: 32 x 24 = {len(frame)} temperature readings")
    print(f"Minimum: {min(frame):.1f} C")
    print(f"Center:  {center:.1f} C")
    print(f"Maximum: {max(frame):.1f} C")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

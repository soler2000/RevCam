# Bill of Materials

| Item | Qty | Notes |
| --- | --- | --- |
| Raspberry Pi Zero 2 W | 1 | Primary compute platform targeted by RevCam deployments. 【F:README.md†L3-L6】 |
| Raspberry Pi camera module (Picamera2-compatible) | 1 | Native Pi camera supported directly through the Picamera2 backend for live video capture. 【F:README.md†L455-L463】 |
| 16-pixel WS2812/NeoPixel LED ring | 1 | Status indicator connected to GPIO18 for boot/ready/error lighting cues. 【F:README.md†L143-L166】 |
| INA219 current/voltage sensor breakout | 1 (optional) | Provides battery telemetry over I²C at address 0x43 for live percentage/voltage/current readouts. 【F:README.md†L183-L218】 |
| VL53L1X time-of-flight distance sensor | 1 (optional) | Supplies distance measurements over I²C at address 0x29 for reversing overlays and geometry calculations. 【F:README.md†L220-L259】 |

> Quantities assume a single RevCam installation. Optional components can be omitted when the corresponding features are not required.

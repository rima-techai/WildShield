# WildShield enclosure concept

Open `WildShield-enclosure.scad` in OpenSCAD for an editable 3D model. Select `part` as `assembly`, `body`, `lid`, or `bracket`. Render and export each part as STL after engineering review. The assembly shows the rear lid separated by 6 mm for visibility.

Open `WildShield-engineering-concept.svg` for the dimensioned concept drawing. All dimensions are millimetres. Proposed enclosure: 120 W × 60 H × 40 D, 3 mm nominal walls. Front openings: RGB Ø20, thermal Ø24, status LED Ø3. Sensor centres are 60 mm apart. These are design assumptions, not measurements of current hardware.

This is a packaging concept, not a fabrication-ready or automotive-qualified design. Confirm the exact RGB module, MLX90640 board, MCP2221A bridge, connectors and cable routing before changing the model into a manufacturing drawing. Current laptop processing remains external; the pictured processing-board space does not establish onboard compute feasibility.

Engineering handoff:
- Replace generic apertures with verified module mounting patterns and field-of-view clearances.
- Keep the thermal optical path open or specify a suitable infrared-transmitting window. Ordinary glass is unsuitable.
- Detail and validate bracket pivots, matching enclosure inserts, fasteners and adjustment range. The bracket is illustrative and not a validated load-bearing joint.
- Specify tolerances, screw engagement, print orientation, material, heat management and strain relief.
- Verify thermal isolation from internal electronics and sensor alignment. No spatial calibration is implied.
- Outdoor mounting requires environmental protection and appropriate placement; no waterproofing or vehicle-installation suitability is established.

Planned deployment options: integration into existing hardware, or a separate sensor device. The separate device is illustrated here. Existing hardware integration needs a host-specific mechanical design.

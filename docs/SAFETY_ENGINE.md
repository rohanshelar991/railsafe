# Safety Engine

The safety engine requires a confirmed, eligible tracked obstacle before automatic braking can occur. It combines estimated TTC with reaction distance, braking distance, total stopping distance, safety margin, and trajectory. A critical result enters virtual emergency braking; the virtual speed then decays until `STOPPED`.

Distance and TTC are estimates. They must be calibrated and compared with ground truth before quantitative safety claims.

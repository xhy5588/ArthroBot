# Joint module

The building block of every ArthroBot robot: one motor (MG4010 in this CAD, MG5010
in simulation) between a 90° holder on its input side and a center connector on its
output side. Carbon rods and printed brackets join modules into limbs.

`cad/single_module/` is the onshape-to-robot export of one module: two links
(`90deg_holder`, the input side, and `center_connecter`, the output side) and one
revolute joint, `motor_joint`. It is reference geometry only. Its masses are
placeholders (1e-9 kg), so no simulation uses it directly.

The arm and the humanoid are separate CAD assemblies of the same parts. Their
builders find each module's two halves by part name:

- motor housing: `mg4010_fc`, which moves with the input side;
- output carrier: `mg4010_pc`, or the output gear on joints where the motor drives
  through a gear, which moves with the output side.

Part masses are shared through `source/arthrobot/data/parts.json` and motor
specifications through `source/arthrobot/data/motors.json`.

Next step (planned): a module and connector library from which new morphologies are
described in YAML instead of a full CAD assembly.

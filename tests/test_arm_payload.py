"""The arm payload calculator (tools/arm_payload): signed statics and the frozen reference.

Runs the tool's own checks (tools/arm_payload/payload_analysis/verify.py) under pytest:
- the signed payload-interval math (counterbalance, impossible vs zero, unbounded);
- ArthroBot's arm model equals, parameter by parameter, the model the included reach
  map was computed with (workspace_payload/model_fingerprint.json);
- the saved reference poses reproduce to 12 digits, and gravity matches the gradient
  of the potential energy.
No simulator or browser is needed. The browser checks are in tools/arm_payload/payload_viewer.
"""
import sys

from arthrobot import paths

sys.path.insert(0, str(paths.REPO_ROOT / 'tools' / 'arm_payload'))

from payload_analysis.verify import OriginalArmRegressionTests, SignedStaticsTests  # noqa: E402,F401

"""Brain MRI cross-scanner domain adaptation configuration.

Used for both IXI (3 scanners) and CC-359 (6 domains).
"""

from medadaptagent.pipeline_config import IXI_CONFIG, CC359_CONFIG

# Re-export for convenience
BRAIN_IXI_CONFIG = IXI_CONFIG
BRAIN_CC359_CONFIG = CC359_CONFIG

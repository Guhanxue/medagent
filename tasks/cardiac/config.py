"""Cardiac cross-modality domain adaptation configuration (CT -> MRI)."""

from medadaptagent.pipeline_config import TaskConfig

CARDIAC_CONFIG = TaskConfig(
    task_name="cardiac_cross_modality",
    source_domain="CT (Hounsfield units, high bone/calcification contrast)",
    target_domain="MRI (balanced SSFP, high soft-tissue contrast)",
    structures={"background": 0, "MYO": 1, "LA": 2, "LV": 3, "AA": 4},
    class_descriptions={
        "MYO": "Myocardium: heart muscle wall, ring-like structure",
        "LA": "Left atrium: upper-left chamber of the heart",
        "LV": "Left ventricle: lower-left pumping chamber, elliptical cavity",
        "AA": "Ascending aorta: large vessel above the heart",
    },
    anatomy_priors={
        "MYO": {"shape": "ring/annular", "area_range": (0.02, 0.15)},
        "LV": {"shape": "elliptical cavity", "area_range": (0.01, 0.10)},
        "LA": {"shape": "irregular chamber", "area_range": (0.01, 0.10)},
        "AA": {"shape": "circular cross-section", "area_range": (0.005, 0.05)},
    },
    containment_rules=[("MYO", "contains", "LV")],
)

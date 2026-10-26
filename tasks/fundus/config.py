"""Fundus cross-device domain adaptation configuration (REFUGE -> RIM-ONE)."""

from medadaptagent.pipeline_config import TaskConfig

FUNDUS_CONFIG = TaskConfig(
    task_name="fundus_cross_device",
    source_domain="REFUGE (Zeiss Visucam 500 + Canon CR-2, full retina FOV)",
    target_domain="RIM-ONE DL (multiple hospitals, various cameras, disc-centered crop)",
    structures={"background": 0, "disc": 1, "cup": 2},
    class_descriptions={
        "disc": "Optic disc: bright elliptical region where optic nerve enters retina",
        "cup": "Optic cup: central depression within the optic disc, paler than disc rim",
    },
    anatomy_priors={
        "disc": {
            "shape": "roughly circular/elliptical",
            "area_range": (0.02, 0.40),
            "description": "Single connected component, 10-40% of cropped image",
        },
        "cup": {
            "shape": "roughly circular, inside disc",
            "area_range": (0.005, 0.25),
            "description": "Single connected component, CDR typically 0.3-0.7",
        },
    },
    containment_rules=[("disc", "contains", "cup")],
)

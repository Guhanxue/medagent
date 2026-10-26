"""CrossMoDA cross-sequence domain adaptation configuration (ceT1 -> hrT2)."""

from medadaptagent.pipeline_config import TaskConfig

CROSSMODA_CONFIG = TaskConfig(
    task_name="crossmoda_cross_sequence",
    source_domain="ceT1 MRI (contrast-enhanced T1, tumor enhances brightly)",
    target_domain="hrT2 MRI (high-resolution T2, tumor appears hyperintense)",
    structures={"background": 0, "VS": 1, "cochlea": 2},
    class_descriptions={
        "VS": "Vestibular schwannoma: tumor at cerebellopontine angle",
        "cochlea": "Cochlea: spiral-shaped inner ear structure",
    },
    anatomy_priors={
        "VS": {"shape": "irregular mass", "area_range": (0.001, 0.10)},
        "cochlea": {"shape": "small spiral/snail-like", "area_range": (0.0005, 0.01)},
    },
    containment_rules=[],
)

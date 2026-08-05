from scripts.run_formal_ablation_variant import VARIANTS, _evaluation_from_frozen
from src.match_engine.calibration.objective import calibration_loss


def test_formal_variant_catalog_covers_declared_ablation_layers():
    assert set(VARIANTS) == {
        "no_affective", "no_spatial", "no_blend", "no_discipline_tick",
        "no_tactical_bias", "M1", "C1",
    }


def test_frozen_report_adapter_preserves_calibration_loss_inputs():
    adapted = _evaluation_from_frozen({
        "report": {"metric": {"z_abs": 1.0, "pass": True}},
        "soft_constraints": {},
        "failed_metrics": [],
        "failed_soft": [],
    })
    assert calibration_loss(adapted) == 1.0

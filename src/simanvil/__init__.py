"""SimAnvil: Deterministic MuJoCo asset forging, contact inspection & VLA dataset assaying."""

from simanvil.assay import ANOMALY_BC_WEIGHT, assay_to_lerobot_v3, compute_episode_telemetry
from simanvil.forge import (
    ForgeConfig,
    MATERIAL_PRESETS,
    attach_forged_asset,
    find_blender_binary,
    forge_asset,
)
from simanvil.inspect import export_inspection_bundle
from simanvil.trace import (
    AnvilRecorder,
    Episode,
    load_episodes,
    register_episode_adapter,
    verify_causal_replay,
)

__version__ = "0.1.0"

__all__ = [
    "ANOMALY_BC_WEIGHT",
    "AnvilRecorder",
    "Episode",
    "ForgeConfig",
    "MATERIAL_PRESETS",
    "assay_to_lerobot_v3",
    "attach_forged_asset",
    "compute_episode_telemetry",
    "export_inspection_bundle",
    "find_blender_binary",
    "forge_asset",
    "load_episodes",
    "register_episode_adapter",
    "verify_causal_replay",
]

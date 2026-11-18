"""Tests for `simanvil.forge`: Z-up dual export, group=2/3/4 MJCF standard, and MjSpec attachment."""

from __future__ import annotations

from pathlib import Path

from lxml import etree
import mujoco as mj

from simanvil import ForgeConfig, attach_forged_asset, forge_asset


def test_forge_z_up_alignment_groups_and_mjspec_attach(tmp_path: Path) -> None:
    asset_dir = tmp_path / "forged_tray"
    report = forge_asset(
        ForgeConfig(
            asset_name="kitting_tray",
            preset="kitting_tray",
            material="aluminum_6061",
            threshold=0.03,
            max_convex_hull=18,
            preprocess_resolution=50,
            output_dir=asset_dir,
        )
    )

    assert report["z_up_alignment_delta_m"] < 1e-4
    assert report["mjcf_groups"] == {"visual": 2, "collision": 3, "sites": 4}
    assert report["smoke_test"]["passed"] is True
    assert report["smoke_test"]["cavity_clearance_gain_mm"] > 24.0

    # Verify standalone asset.xml uses group="2" visual, group="3" collision, group="4" sites
    xml_doc = etree.parse(str(asset_dir / "asset.xml"))
    vis_geoms = xml_doc.xpath("//geom[@group='2']")
    col_geoms = xml_doc.xpath("//geom[@group='3']")
    sites = xml_doc.xpath("//site[@group='4']")
    assert len(vis_geoms) == 1
    assert len(col_geoms) == report["hull_count"]
    assert len(sites) >= 2

    # Verify direct attachment onto any MjSpec scene
    spec = mj.MjSpec()
    spec.worldbody.add_geom(name="ground", type=mj.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01])
    attach_forged_asset(spec, asset_dir, pos=(0.2, 0.0, 0.04), dynamic=True, mode="coacd")
    compiled = spec.compile()
    assert compiled.body("kitting_tray").id > 0
    assert compiled.site("kitting_tray_origin").id >= 0

"""Unified CLI for SimAnvil (`simanvil forge | inspect | assay | serve`)."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from simanvil.assay.lerobot_v3 import assay_to_lerobot_v3
from simanvil.forge.coacd_compiler import ForgeConfig, MATERIAL_PRESETS, forge_asset
from simanvil.inspect.bundle_exporter import export_inspection_bundle
from simanvil.trace.adapters import load_episodes
from simanvil.trace.schema import Episode


def _serve_directory(directory: Path, host: str, port: int) -> None:
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(f"No index.html found in {directory}")
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"SimAnvil Session Viewer serving at http://{host}:{port}/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def cmd_forge(args: argparse.Namespace) -> None:
    cfg = ForgeConfig(
        asset_name=args.name,
        output_dir=args.output,
        input_mesh=args.input,
        preset=args.preset,
        material=args.material,
        threshold=args.threshold,
        max_convex_hull=args.max_hulls,
        preprocess_resolution=args.resolution,
    )
    report = forge_asset(cfg)
    print(
        json.dumps(
            {
                "asset_name": report["asset_name"],
                "mjcf_groups": report["mjcf_groups"],
                "hull_count": report["hull_count"],
                "z_up_alignment_delta_m": report["z_up_alignment_delta_m"],
                "freed_void_cm3": report["occluded_void_volume_cm3"],
                "smoke_test": report["smoke_test"],
                "output_dir": str(args.output.resolve()),
            },
            indent=2,
        )
    )


def cmd_inspect(args: argparse.Namespace) -> None:
    episodes: list[Episode] = []
    for tgt in args.targets:
        episodes.extend(
            load_episodes(
                tgt,
                model_xml=args.model,
                max_frames=args.max_frames,
                render_cameras=True,
            )
        )

    out_dir = args.output.resolve()
    summary = export_inspection_bundle(output_dir=out_dir, episodes=episodes, title=args.title)

    if args.assay_out is not None:
        lr_manifest = assay_to_lerobot_v3(
            episodes=episodes,
            output_dir=args.assay_out,
            repo_id=args.repo_id,
            anomaly_policy=args.anomaly_policy,
        )
        summary["lerobot_v3"] = {
            "dataset_dir": lr_manifest["dataset_dir"],
            "episodes_exported": lr_manifest["episodes_exported"],
            "total_frames": lr_manifest["total_frames"],
            "clean_frames_1x": lr_manifest["clean_frames_1x"],
            "downweighted_frames": lr_manifest["downweighted_frames"],
            "effective_weighted_frames": lr_manifest["effective_weighted_frames"],
            "max_decoded_rgb_mae": lr_manifest["max_decoded_rgb_mae_0_to_1"],
        }

    print(json.dumps(summary, indent=2))
    if args.serve:
        _serve_directory(out_dir, args.host, args.port)


def cmd_assay(args: argparse.Namespace) -> None:
    episodes: list[Episode] = []
    for tgt in args.targets:
        episodes.extend(
            load_episodes(
                tgt,
                model_xml=args.model,
                max_frames=args.max_frames,
                render_cameras=True,
            )
        )
    manifest = assay_to_lerobot_v3(
        episodes=episodes,
        output_dir=args.output,
        repo_id=args.repo_id,
        anomaly_policy=args.anomaly_policy,
    )
    print(json.dumps(manifest, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    _serve_directory(args.dir.resolve(), args.host, args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="simanvil", description="SimAnvil v0.1 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_forge = sub.add_parser("forge", help="Forge a 3D mesh or preset via Headless Blender + CoACD + Smoke Audit")
    p_forge.add_argument("--name", required=True, help="Logical body/asset name in MJCF")
    grp = p_forge.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input", type=Path, help="External mesh file (.glb/.gltf/.obj/.stl/.ply)")
    grp.add_argument("--preset", choices=["kitting_tray", "u_bracket"], help="Built-in concave test preset")
    p_forge.add_argument("--output", type=Path, required=True, help="Output directory for forged bundle")
    p_forge.add_argument("--material", choices=sorted(MATERIAL_PRESETS.keys()), default="aluminum_6061")
    p_forge.add_argument("--threshold", type=float, default=0.02, help="CoACD concavity threshold")
    p_forge.add_argument("--max-hulls", type=int, default=24, help="Maximum convex hulls per part")
    p_forge.add_argument("--resolution", type=int, default=65, help="CoACD voxel preprocess resolution")
    p_forge.set_defaults(func=cmd_forge)

    p_ins = sub.add_parser("inspect", help="Build a static HTML 3D Session Viewer from MuJoCo episode(s)")
    p_ins.add_argument("targets", nargs="+", type=Path, help="Episode .npz file(s) or run directories")
    p_ins.add_argument("--model", type=Path, default=None, help="Optional MJCF scene.xml if not alongside .npz")
    p_ins.add_argument("--output", type=Path, default=Path("outputs/session"))
    p_ins.add_argument("--title", type=str, default=None)
    p_ins.add_argument("--max-frames", type=int, default=220, help="Max sampled frames per episode (0 = all)")
    p_ins.add_argument("--assay-out", type=Path, default=None, help="Optionally export weighted LeRobot v3 dataset")
    p_ins.add_argument("--repo-id", type=str, default="local/simanvil_dataset")
    p_ins.add_argument(
        "--anomaly-policy",
        choices=["downweight", "zero-anomaly-frames", "drop-failed-episodes"],
        default="downweight",
    )
    p_ins.add_argument("--serve", action="store_true")
    p_ins.add_argument("--host", default="127.0.0.1")
    p_ins.add_argument("--port", type=int, default=8895)
    p_ins.set_defaults(func=cmd_inspect)

    p_assay = sub.add_parser("assay", help="Assay episode(s) into a quality-weighted LeRobot v3 dataset")
    p_assay.add_argument("targets", nargs="+", type=Path)
    p_assay.add_argument("--model", type=Path, default=None)
    p_assay.add_argument("--output", type=Path, required=True)
    p_assay.add_argument("--repo-id", type=str, default="local/simanvil_dataset")
    p_assay.add_argument("--max-frames", type=int, default=0, help="0 preserves all native frames & substeps")
    p_assay.add_argument(
        "--anomaly-policy",
        choices=["downweight", "zero-anomaly-frames", "drop-failed-episodes"],
        default="downweight",
    )
    p_assay.set_defaults(func=cmd_assay)

    p_srv = sub.add_parser("serve", help="Serve an exported SimAnvil HTML Session Viewer directory")
    p_srv.add_argument("--dir", type=Path, default=Path("outputs/session"))
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8895)
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

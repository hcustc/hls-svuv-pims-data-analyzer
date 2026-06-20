from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bl03u_masstool.core.analysis_artifacts import (
    build_analysis_manifest,
    build_pie_evidence_objects,
    build_temperature_evidence_objects,
    write_artifact_bundle,
)
from bl03u_masstool.core.config import load_calibration_config, load_peak_detection_config, project_path
from bl03u_masstool.core.isotope import (
    calculate_isotope_distribution,
    formula_monoisotopic_mass,
    formula_nominal_mass,
    parse_formula,
)
from bl03u_masstool.core.normalization import load_normalization_settings
from bl03u_masstool.core.pie_analysis import (
    analyze_pie_folder,
    build_pie_curves,
    identify_species_for_mz_with_curve,
    load_species_database,
)
from bl03u_masstool.core.temperature_scan import analyze_temperature_folder, build_temperature_curves


def _parse_mz_values(value: str | None) -> list[int] | None:
    if not value:
        return None
    values: list[int] = []
    for token in re.split(r"[,，;；\s]+", value.strip()):
        if not token:
            continue
        values.append(int(round(float(token))))
    return sorted(set(values))


def _resolve_output(path: str | Path) -> Path:
    output = project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _curve_json_ready(curves: dict[int, dict]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for mz, curve in curves.items():
        ready[str(mz)] = {
            "mz": int(curve["mz"]),
            "mz_exact_mean": float(curve.get("mz_exact_mean", curve["mz"])),
            "species": curve.get("species", ""),
            "energies": [float(value) for value in curve.get("energies", [])],
            "intensities": [float(value) for value in curve.get("intensities", [])],
            "temperatures": [float(value) for value in curve.get("temperatures", [])],
            "areas": [float(value) for value in curve.get("areas", [])],
            "curve_class": curve.get("curve_class"),
            "curve_class_label": curve.get("curve_class_label"),
            "curve_class_reason": curve.get("curve_class_reason"),
        }
    return ready


def _write_json(path: str | Path, payload: Any) -> Path:
    output = _resolve_output(path)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output


def _peak_kwargs() -> dict[str, Any]:
    return load_peak_detection_config().to_peak_kwargs()


def cmd_pie(args: argparse.Namespace) -> int:
    normalization = load_normalization_settings()
    photon_mode = args.photon_mode or normalization.pie_photon_mode
    light_source = args.light_source or normalization.light_source
    mass_discrimination = (
        args.mass_discrimination
        if args.mass_discrimination is not None
        else normalization.mass_discrimination
    )
    df = analyze_pie_folder(
        args.folder,
        calibration=load_calibration_config(),
        recursive=not args.no_recursive,
        energy_decimals=args.energy_decimals,
        prefer_gaussian=not args.no_gaussian,
        manual_peak_path=args.manual_peak_path,
        photon_normalize=photon_mode != "off",
        photon_reference_mode="none" if photon_mode == "off" else photon_mode,
        mass_discrimination=mass_discrimination,
        light_source=light_source,
        target_mz_values=_parse_mz_values(args.target_mz),
        **_peak_kwargs(),
    )
    output = _resolve_output(args.output)
    df.to_csv(output, index=False)
    curves = build_pie_curves(df)
    outputs = {"csv": output}
    if args.curves_json:
        curves_path = _write_json(args.curves_json, _curve_json_ready(curves))
        outputs["curves_json"] = curves_path

    fits: dict[int, dict[str, Any]] = {}
    if args.fit_pics:
        database, _ = load_species_database(args.database)
        for mz, curve in curves.items():
            fits[int(mz)] = identify_species_for_mz_with_curve(
                database,
                int(mz),
                curve.get("energies", []),
                curve.get("intensities", []),
            )

    if args.manifest_json or args.evidence_json or args.report_md:
        manifest = build_analysis_manifest(
            analysis_type="pie",
            input_path=args.folder,
            parameters={
                "recursive": not args.no_recursive,
                "energy_decimals": args.energy_decimals,
                "prefer_gaussian": not args.no_gaussian,
                "manual_peak_path": args.manual_peak_path,
                "target_mz": args.target_mz,
                "photon_mode": photon_mode,
                "light_source": light_source,
                "mass_discrimination": mass_discrimination,
                "fit_pics": bool(args.fit_pics),
                "database": args.database,
            },
            outputs={
                **outputs,
                "manifest_json": args.manifest_json,
                "evidence_json": args.evidence_json,
                "report_md": args.report_md,
            },
            analysis_df=df,
            curves=curves,
        )
        evidence = build_pie_evidence_objects(curves, fits)
        write_artifact_bundle(
            manifest=manifest,
            evidence=evidence,
            manifest_json=_resolve_output(args.manifest_json) if args.manifest_json else None,
            evidence_json=_resolve_output(args.evidence_json) if args.evidence_json else None,
            report_md=_resolve_output(args.report_md) if args.report_md else None,
        )
    print(f"wrote {len(df)} rows and {len(curves)} curves to {output}")
    return 0


def cmd_temperature(args: argparse.Namespace) -> int:
    normalization = load_normalization_settings()
    light_source = args.light_source or normalization.light_source
    mass_discrimination = (
        args.mass_discrimination
        if args.mass_discrimination is not None
        else normalization.mass_discrimination
    )
    df = analyze_temperature_folder(
        args.folder,
        calibration=load_calibration_config(),
        prefer_gaussian=not args.no_gaussian,
        reference_mode=args.reference_mode,
        manual_peak_path=args.manual_peak_path,
        photon_normalize=not args.no_photon_normalize,
        kr_correct=args.kr_correct,
        kr_mz=args.kr_mz,
        mass_discrimination=mass_discrimination,
        light_source=light_source,
        **_peak_kwargs(),
    )
    output = _resolve_output(args.output)
    df.to_csv(output, index=False)
    curves = build_temperature_curves(df)
    outputs = {"csv": output}
    if args.curves_json:
        curves_path = _write_json(args.curves_json, _curve_json_ready(curves))
        outputs["curves_json"] = curves_path
    if args.manifest_json or args.evidence_json or args.report_md:
        manifest = build_analysis_manifest(
            analysis_type="temperature",
            input_path=args.folder,
            parameters={
                "reference_mode": args.reference_mode,
                "prefer_gaussian": not args.no_gaussian,
                "manual_peak_path": args.manual_peak_path,
                "photon_normalize": not args.no_photon_normalize,
                "kr_correct": args.kr_correct,
                "kr_mz": args.kr_mz,
                "light_source": light_source,
                "mass_discrimination": mass_discrimination,
            },
            outputs={
                **outputs,
                "manifest_json": args.manifest_json,
                "evidence_json": args.evidence_json,
                "report_md": args.report_md,
            },
            analysis_df=df,
            curves=curves,
        )
        evidence = build_temperature_evidence_objects(curves)
        write_artifact_bundle(
            manifest=manifest,
            evidence=evidence,
            manifest_json=_resolve_output(args.manifest_json) if args.manifest_json else None,
            evidence_json=_resolve_output(args.evidence_json) if args.evidence_json else None,
            report_md=_resolve_output(args.report_md) if args.report_md else None,
        )
    print(f"wrote {len(df)} rows and {len(curves)} curves to {output}")
    return 0


def cmd_formula(args: argparse.Namespace) -> int:
    payload = {
        "formula": args.formula,
        "composition": parse_formula(args.formula),
        "nominal_mass": formula_nominal_mass(args.formula),
        "monoisotopic_mass": formula_monoisotopic_mass(args.formula),
    }
    if args.isotopes:
        payload["isotope_distribution"] = calculate_isotope_distribution(
            args.formula,
            min_percent=args.min_percent,
        )
    if args.output:
        _write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BL03U MassSpectrumTool batch CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pie = subparsers.add_parser("pie", help="Generate PIE curves from an energy scan folder")
    pie.add_argument("folder")
    pie.add_argument("-o", "--output", default="output/exports/pie_analysis.csv")
    pie.add_argument("--curves-json")
    pie.add_argument("--manual-peak-path")
    pie.add_argument("--target-mz")
    pie.add_argument("--no-recursive", action="store_true")
    pie.add_argument("--energy-decimals", type=int, default=1)
    pie.add_argument("--no-gaussian", action="store_true")
    pie.add_argument("--photon-mode", choices=["first", "none", "off"])
    pie.add_argument("--light-source", choices=["io", "beam_current"])
    pie.add_argument("--mass-discrimination", type=float)
    pie.add_argument("--manifest-json")
    pie.add_argument("--evidence-json")
    pie.add_argument("--report-md")
    pie.add_argument("--fit-pics", action="store_true")
    pie.add_argument("--database")
    pie.set_defaults(func=cmd_pie)

    temperature = subparsers.add_parser("temperature", help="Analyze a temperature scan folder")
    temperature.add_argument("folder")
    temperature.add_argument("-o", "--output", default="output/exports/temperature_analysis.csv")
    temperature.add_argument("--curves-json")
    temperature.add_argument("--manual-peak-path")
    temperature.add_argument("--reference-mode", choices=["sum", "max_temperature"], default="sum")
    temperature.add_argument("--no-gaussian", action="store_true")
    temperature.add_argument("--no-photon-normalize", action="store_true")
    temperature.add_argument("--kr-correct", action="store_true")
    temperature.add_argument("--kr-mz", type=int, default=84)
    temperature.add_argument("--light-source", choices=["io", "beam_current"])
    temperature.add_argument("--mass-discrimination", type=float)
    temperature.add_argument("--manifest-json")
    temperature.add_argument("--evidence-json")
    temperature.add_argument("--report-md")
    temperature.set_defaults(func=cmd_temperature)

    formula = subparsers.add_parser("formula", help="Parse a formula and calculate masses")
    formula.add_argument("formula")
    formula.add_argument("-o", "--output")
    formula.add_argument("--isotopes", action="store_true")
    formula.add_argument("--min-percent", type=float, default=0.01)
    formula.set_defaults(func=cmd_formula)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pie" and args.fit_pics and not args.database:
        parser.error("--fit-pics requires --database")
    if args.command == "optimize-peaks" and args.fit_pics and not args.database:
        parser.error("--fit-pics requires --database")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

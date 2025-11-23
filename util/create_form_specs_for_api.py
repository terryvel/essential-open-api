"""
Generate form specifications for the Essential Open API forms.

This script is intended to run from the command line. It reads a .pprj file,
extracts form widgets, applies filtering/grouping, and writes one JSON file per
class into the target directory (creating it if necessary). Files are written
only once—classes that do not meet filtering criteria are skipped without
temporary files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_PPRJ = Path("../resources/essential_baseline_6_20.pprj")
DEFAULT_OUTPUT_DIR = Path("../resources/forms")


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_id(raw) -> str:
    return str(raw).strip().strip("[]")


def parse_pprj_file(pprj_path: Path) -> Dict[str, List[dict]]:
    """Parse the .pprj file and return raw form fields per class."""
    content = pprj_path.read_text(encoding="utf-8")

    # Extract all blocks: ([id] of ClassName (...) )
    blocks = re.findall(r"\(\[([^\]]+)\] of\s+([^\s]+)\s*\((.*?)\)\)", content, re.DOTALL)

    widgets: Dict[str, dict] = {}
    property_lists: Dict[str, dict] = {}
    forms: Dict[str, dict] = {}

    for block_id, block_type, block_body in blocks:
        if not block_body.endswith(")"):
            block_body = f"{block_body})"

        lines = re.findall(r"\(([^)]+)\)", block_body)
        props: Dict[str, object] = {}
        for line in lines:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            key, val = parts
            if key == "properties":
                prop_ids = re.findall(r"\[([^\]]+)\]", val)
                props[key] = prop_ids
            else:
                props[key] = val.strip('"')

        if block_type == "Widget":
            widgets[block_id] = props
            if props.get("widget_class_name", "").endswith("FormWidget"):
                forms[block_id] = props
        elif block_type == "Property_List":
            prop_ids = re.findall(r"\[([^\]]+)\]", block_body)
            property_lists[block_id] = {"properties": prop_ids}
        elif block_type == "FormWidget":
            forms[block_id] = props

    form_fields: Dict[str, List[dict]] = {}
    for form_id, form_props in forms.items():
        class_name = form_props.get("name")
        prop_list_id = _normalize_id(form_props.get("property_list"))
        if not class_name or not prop_list_id:
            continue

        properties = property_lists.get(prop_list_id, {}).get("properties", [])

        fields = []
        for i, widget_id in enumerate(properties):
            widget = widgets.get(widget_id)
            if not widget:
                continue

            fields.append(
                {
                    "slot": widget.get("name", ""),
                    "label": widget.get("label", widget.get("name", "")),
                    "widget_type": widget.get("widget_class_name", "").split(".")[-1],
                    "order": i + 1,
                    "x": _to_int(widget.get("x", 0)),
                    "y": _to_int(widget.get("y", 0)),
                    "width": _to_int(widget.get("width", 0)),
                    "height": _to_int(widget.get("height", 0)),
                    "is_hidden": str(widget.get("is_hidden", "FALSE")).upper() == "TRUE",
                }
            )

        form_fields[class_name] = fields

    return form_fields


def refine_fields(fields: List[dict]) -> Optional[List[dict]]:
    """Filter and order fields. Return None if the form should be skipped."""
    valid = [f for f in fields if f.get("widget_type", "").strip()]
    labels = [f.get("label", "").strip() for f in valid]

    if not valid:
        return None
    if "Classified As" not in labels:
        return None

    ordered = sorted(valid, key=lambda w: (_to_int(w.get("y", 0)), _to_int(w.get("x", 0))))

    cleaned = []
    for i, field in enumerate(ordered, start=1):
        cleaned.append(
            {
                "slot": field.get("slot", ""),
                "label": field.get("label", ""),
                "widget_type": field.get("widget_type", ""),
                "order": i,
            }
        )
    return cleaned


def classify_groups(fields: List[dict]) -> List[dict]:
    """Assign groups to fields and return grouped structure."""
    fixed_groups = ["principal", "extra", "security", "system"]

    has_supersedes = any(f.get("slot") == "supersedes_version" for f in fields)
    base_slots = []
    if has_supersedes:
        for f in fields:
            if f.get("slot") == "supersedes_version":
                break
            base_slots.append(f.get("slot"))
    else:
        base_slots = ["name", "description"]

    for f in fields:
        slot = f.get("slot", "")
        if slot in base_slots:
            f["group"] = "principal"
        elif slot.startswith("system_security"):
            f["group"] = "security"
        elif slot.startswith("system"):
            f["group"] = "system"
        else:
            f["group"] = "extra"

    grouped = []
    for group in fixed_groups:
        group_items = [
            {
                "slot": f["slot"],
                "label": f["label"],
                "widget_type": f["widget_type"],
                "order": f["order"],
            }
            for f in fields
            if f.get("group") == group
        ]
        if group_items:
            grouped.append({"group": group, "items": group_items})

    return grouped


def generate_form_specs(pprj_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    form_fields = parse_pprj_file(pprj_path)
    written = 0
    skipped = 0

    for class_name, fields in form_fields.items():
        cleaned = refine_fields(fields)
        if cleaned is None:
            skipped += 1
            continue

        grouped = classify_groups(cleaned)
        output_path = output_dir / f"{class_name}.json"
        output_path.write_text(
            json.dumps({class_name: grouped}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written += 1
        print(f"Generated: {output_path}")

    print(f"Done. Written: {written}, Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate form specs for Essential Open API.",
        epilog=(
            "Examples:\n"
            "  python util/create_form_specs_for_api.py\n"
            "  python util/create_form_specs_for_api.py --pprj ../resources/essential_baseline_6_20.pprj "
            "--out ../resources/forms_custom"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pprj",
        type=Path,
        default=DEFAULT_PPRJ,
        help=f"Path to the .pprj file (default: {DEFAULT_PPRJ})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for JSON files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    generate_form_specs(args.pprj, args.out)


if __name__ == "__main__":
    main()

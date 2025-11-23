"""Blueprint with API endpoints for the Essential Open API."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import List, Optional
from pathlib import Path

from flask import Blueprint, jsonify, request

from .jvm import (
    call_publish_async,
    get_knowledge_base,
    get_project,
    get_publish_status,
    save_project,
)

api_bp = Blueprint("api", __name__)
SAFE_FRAME_NAME_RE = re.compile(r"[^0-9A-Za-z_]")
RESERVED_FIELDS = {"className", "name", "description", "externalId"}
NAME_SLOT_CANDIDATES = ("name_", "name", "relation_name", ":relation_name")


class InstanceCreationError(Exception):
    """Custom exception to propagate creation errors with HTTP status codes."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class CreatedInstance:
    """Container for instance metadata returned by creation helpers."""

    instance: object
    description: Optional[str]
    external_summary: Optional[dict]


@api_bp.get("/list_items")
def list_instances():
    """List all instances of a given class from the Protégé knowledge base."""
    kb = get_knowledge_base()
    if kb is None:
        return jsonify({"error": "Knowledge Base not loaded!"}), 500

    class_name = request.args.get("class")
    if not class_name:
        return jsonify({"error": "Parameter 'class' is required!"}), 400

    cls = kb.getCls(class_name)
    if not cls:
        return jsonify({"error": f"Class '{class_name}' not found!"}), 404

    instances = [str(inst.getBrowserText()) for inst in cls.getInstances()]

    return jsonify({"class": class_name, "instances": instances})


@api_bp.get("/publish")
def publish():
    """Trigger asynchronous publish process using the loaded project."""
    project = get_project()
    if project is None:
        return jsonify({"error": "Project not loaded!"}), 500

    data = request.get_json(silent=True)
    if data:
        url = data.get("url")
        user = data.get("user")
        pwd = data.get("pwd")
    else:
        url = request.args.get("url")
        user = request.args.get("user")
        pwd = request.args.get("pwd")

    url = url or "http://host.docker.internal:9090/essential_viewer"
    user = user or "alice"
    pwd = pwd or "s3cr3t"

    job_id = call_publish_async(project, url, user, pwd)
    if job_id is None:
        return jsonify({"error": "Failed to start publish job."}), 500

    return jsonify({"jobId": job_id})


@api_bp.get("/publish-status")
def publish_status():
    """Return the status and logs for a given publication job."""
    job_id = request.args.get("id")
    if not job_id:
        return jsonify({"error": "Parameter 'id' is required"}), 400

    result = get_publish_status(job_id)
    if result is None:
        return jsonify({"error": "Unable to fetch publish status."}), 500

    status, logs = result
    return jsonify({"jobId": job_id, "status": status, "logs": logs})


@api_bp.get("/classes/")
def list_root_classes():
    """Return the list of classes at the root (without superclasses)."""
    kb = get_knowledge_base()
    if kb is None:
        return jsonify({"error": "Knowledge Base not loaded!"}), 500

    try:
        classes = kb.getClses() or []
    except Exception as exc:
        return jsonify({"error": f"Failed to enumerate classes: {exc}"}), 500

    root_classes = []
    for cls in classes:
        try:
            supers = cls.getDirectSuperclasses() or []
        except Exception:
            continue
        if supers:
            continue

        try:
            title = str(cls.getBrowserText())
        except Exception:
            title = ""
        try:
            subclass_count = int(cls.getDirectSubclassCount())
        except Exception:
            subclass_count = 0
        try:
            abstract_flag = bool(cls.isAbstract())
        except Exception:
            abstract_flag = False

        root_classes.append(
            {
                "id": str(cls.getName()),
                "title": title,
                "hasChildren": bool(subclass_count),
                "count": len(cls.getDirectInstances()),
                "isAbstract": abstract_flag,
            }
        )

    try:
        root_cls = getattr(kb, "getRootCls", lambda: None)() or kb.getCls(":THING")
    except Exception:
        root_cls = None
    current_name = str(root_cls.getName()) if root_cls else ""
    try:
        current_is_abstract = bool(root_cls.isAbstract()) if root_cls else False
    except Exception:
        current_is_abstract = False
    try:
        current_count = len(root_cls.getDirectInstances()) if root_cls else 0
    except Exception:
        current_count = 0

    return jsonify({"class": current_name, "isAbstract": current_is_abstract, "count": current_count, "classes": root_classes})


@api_bp.get("/classes/<string:class_name>/")
def list_child_classes(class_name: str):
    """Return direct subclasses for a given class."""
    kb = get_knowledge_base()
    if kb is None:
        return jsonify({"error": "Knowledge Base not loaded!"}), 500

    cls = kb.getCls(class_name)
    if not cls:
        return jsonify({"error": f"Class '{class_name}' not found!"}), 404

    try:
        subclasses = cls.getDirectSubclasses() or []
    except Exception as exc:
        return jsonify({"error": f"Failed to enumerate subclasses: {exc}"}), 500

    children = []
    for subcls in subclasses:
        try:
            title = str(subcls.getBrowserText())
        except Exception:
            title = ""
        try:
            subclass_count = int(subcls.getDirectSubclassCount())
        except Exception:
            subclass_count = 0
        try:
            abstract_flag = bool(subcls.isAbstract())
        except Exception:
            abstract_flag = False
        children.append(
            {
                "id": str(subcls.getName()),
                "title": title,
                "hasChildren": bool(subclass_count),
                "count": len(subcls.getDirectInstances()),
                "isAbstract": abstract_flag,
            }
        )

    try:
        current_cls = kb.getCls(class_name)
    except Exception:
        current_cls = None
    current_name = str(current_cls.getName()) if current_cls else class_name
    try:
        current_is_abstract = bool(current_cls.isAbstract()) if current_cls else False
    except Exception:
        current_is_abstract = False
    try:
        current_count = len(current_cls.getDirectInstances()) if current_cls else 0
    except Exception:
        current_count = 0

    return jsonify({"class": current_name, "isAbstract": current_is_abstract, "count": current_count, "classes": children})

@api_bp.get("/classes/<string:class_name>/form")
def get_class_form(class_name: str):
    """Return the stored form specification JSON for the given class."""
    forms_dir = Path(__file__).resolve().parent.parent / "resources" / "forms"
    form_path = forms_dir / f"{class_name}.json"

    if not form_path.exists():
        fallback = [
            {"group": "principal", "items": []},
            {"group": "extra", "items": []},
            {"group": "security", "items": []},
            {"group": "system", "items": []},
        ]
        return jsonify({class_name: fallback})

    try:
        data = json.loads(form_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"error": f"Failed to load form for '{class_name}': {exc}"}), 500

    return jsonify(data)

@api_bp.get("/classes/<string:class_name>/slots")
def list_class_slots(class_name: str):
    """Return template slots for the given class."""
    kb = get_knowledge_base()
    if kb is None:
        return jsonify({"error": "Knowledge Base not loaded!"}), 500

    cls = kb.getCls(class_name)
    if not cls:
        return jsonify({"error": f"Class '{class_name}' not found!"}), 404

    try:
        template_slots = cls.getTemplateSlots() or []
    except Exception as exc:
        return jsonify({"error": f"Failed to fetch template slots: {exc}"}), 500

    resultados = []
    for slot in template_slots:
        try:
            name = str(slot.getName())
        except Exception:
            name = ""

        try:
            value_type = str(slot.getValueType())
        except Exception:
            value_type = ""

        try:
            min_cardinality = int(cls.getTemplateSlotMinimumCardinality(slot))
        except Exception:
            min_cardinality = None

        try:
            max_cardinality = int(cls.getTemplateSlotMaximumCardinality(slot))
        except Exception:
            max_cardinality = None

        try:
            label = slot.getDirectBrowserText()
        except Exception:
            label = None

        if not label:
            try:
                label_slot = cls.getKnowledgeBase().getSlot("label")
            except Exception:
                label_slot = None
            if label_slot:
                try:
                    label_value = slot.getOwnSlotValue(label_slot)
                    label = str(label_value) if label_value else name
                except Exception:
                    label = name
            else:
                label = name

        resultados.append(
            {
                "slot": name,
                "label": str(label),
                "type": value_type,
                "min_cardinality": min_cardinality,
                "max_cardinality": max_cardinality,
                **(
                    {
                        "allowed_classes": [
                            str(c.getName()) for c in (slot.getAllowedClses() or []) if c
                        ]
                    }
                    if value_type.lower() == "instance"
                    else {}
                ),
            }
        )

    return jsonify(resultados)

# ----- Essential Utility API -----

MAX_DEPTH_DEFAULT = 1

def load_json_body():
    """Load JSON body even if the client forgot the correct header."""
    data = request.get_json(silent=True)
    if data is not None:
        return data, None

    raw = request.get_data(cache=False, as_text=False)
    if not raw:
        return None, "JSON body is required."

    try:
        encoding = request.charset or "utf-8"
    except Exception:
        encoding = "utf-8"

    try:
        return json.loads(raw.decode(encoding)), None
    except Exception:
        return None, "Malformed JSON body."

def sanitize_frame_name(raw: str, fallback: str) -> str:
    """Create a safe Protégé frame identifier."""
    candidate = SAFE_FRAME_NAME_RE.sub("_", (raw or "").strip())
    candidate = candidate.strip("_")
    fallback_clean = SAFE_FRAME_NAME_RE.sub("_", (fallback or "instance"))
    return candidate or fallback_clean or "instance"


def frame_exists(kb, frame_name: str) -> bool:
    """Return True if a frame with the given name already exists."""
    if not frame_name:
        return False
    try:
        getter = getattr(kb, "getFrame", None)
        if callable(getter):
            return getter(frame_name) is not None
    except Exception:
        pass
    try:
        return kb.getInstance(frame_name) is not None
    except Exception:
        return False


def unique_frame_name(kb, preferred: str, class_name: str) -> str:
    """Generate an available frame name."""
    base = sanitize_frame_name(preferred, class_name)
    candidate = base
    suffix = 1
    while frame_exists(kb, candidate):
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def get_slot(kb, slot_name: str):
    """Safely fetch a slot from the knowledge base."""
    if not slot_name:
        return None
    try:
        return kb.getSlot(slot_name)
    except Exception:
        return None


def resolve_slot(obj, kb=None):
    """Return a Slot object whether a slot or slot name is provided."""
    if hasattr(obj, "getName"):
        return obj
    target_kb = kb
    if target_kb is None and obj is not None:
        target_kb = obj.getKnowledgeBase()
    return get_slot(target_kb, obj)


def set_slot_value(inst, slot, value) -> bool:
    """Set the value of a slot if both slot and value exist."""
    if value is None:
        return False
    slot_obj = resolve_slot(slot, inst.getKnowledgeBase())
    if slot_obj is None:
        return False
    try:
        inst.setOwnSlotValue(slot_obj, value)
        return True
    except Exception:
        return False


def add_slot_value(inst, slot, value) -> bool:
    """Append a value to a multi-slot."""
    if value is None:
        return False
    slot_obj = resolve_slot(slot, inst.getKnowledgeBase())
    if slot_obj is None:
        return False
    try:
        inst.addOwnSlotValue(slot_obj, value)
        return True
    except Exception:
        return False


def slot_allows_multiple(slot) -> bool:
    """Return True if the slot accepts multiple values."""
    slot_obj = resolve_slot(slot)
    if slot_obj is None:
        return True
    try:
        return bool(slot_obj.getAllowsMultipleValues())
    except Exception:
        return True


def find_instance_by_slot_value(cls, slot_name: str, expected: str):
    """Return the first instance where slot == expected."""
    if not expected:
        return None
    kb = cls.getKnowledgeBase()
    slot = get_slot(kb, slot_name)
    if slot is None:
        return None
    try:
        instances = cls.getInstances() or []
    except Exception:
        return None
    for inst in instances:
        try:
            current = inst.getOwnSlotValue(slot)
        except Exception:
            continue
        if current and str(current).strip() == expected:
            return inst
    return None


def set_instance_name(instance, name: str) -> bool:
    """Set the most appropriate slot for instance name."""
    kb = instance.getKnowledgeBase()
    for slot_name in NAME_SLOT_CANDIDATES:
        slot = get_slot(kb, slot_name)
        if slot is None:
            continue
        if set_slot_value(instance, slot, name):
            return True

    try:
        instance.setBrowserText(name)
        return True
    except Exception:
        return False


def ensure_external_repository(kb, source_name: str):
    """Find or create an External_Repository instance for the given source."""
    repo_cls = kb.getCls("External_Repository")
    if repo_cls is None:
        return None, "Class 'External_Repository' not found in Knowledge Base."

    existing = find_instance_by_slot_value(repo_cls, "name_", source_name)
    if existing is not None:
        return existing, None

    try:
        repo = kb.createInstance(None, repo_cls)
    except Exception as exc:
        return None, f"Unable to create External Repository: {exc}"

    if not set_instance_name(repo, source_name):
        return None, "Failed to set repository name."
    return repo, None


def create_external_reference_record(kb, target_instance, repo_instance, external_id: str, source_name: str):
    """Create and attach an External_Instance_Reference."""
    ref_cls = kb.getCls("External_Instance_Reference")
    if ref_cls is None:
        return None, "Class 'External_Instance_Reference' not found in Knowledge Base."

    try:
        ref_inst = kb.createInstance(None, ref_cls)
    except Exception as exc:
        return None, f"Unable to create External Instance Reference: {exc}"

    label = f"{source_name}:{external_id}"
    timestamp = datetime.now(timezone.utc).isoformat()
    required_slots = [
        (ref_inst, "external_instance_reference", external_id),
        (ref_inst, "external_update_date", timestamp),
        (ref_inst, "external_repository_reference", repo_instance),
        (ref_inst, "referenced_instance", target_instance),
    ]
    if not set_instance_name(ref_inst, label):
        return None, "Failed to set external reference name."
    for inst_obj, slot_name, value in required_slots:
        if not set_slot_value(inst_obj, slot_name, value):
            return None, f"Failed to set slot '{slot_name}' on external reference."

    return ref_inst, None


def validate_external_id(payload: Optional[dict]) -> Optional[dict]:
    """Validate and normalise the externalId payload."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Field 'externalId' must be an object.")

    source_name = str(payload.get("sourceName", "")).strip()
    identifier = str(payload.get("id", "")).strip()
    if not source_name or not identifier:
        raise ValueError("Fields 'externalId.sourceName' and 'externalId.id' are required.")

    return {"sourceName": source_name, "id": identifier}


def is_frame(v):
    return hasattr(v, "getDirectType") and hasattr(v, "getName")

def normalize_primitive(v):
    return "" if v is None else str(v)

def get_id_name_class(inst):
    try:
        cls_obj = inst.getDirectType()
        class_name = str(cls_obj.getName()) if cls_obj else ""
    except Exception:
        class_name = ""
    return {
        "id": str(inst.getName()),
        "name": str(inst.getBrowserText()),
        "className": class_name,
    }

def iter_filled_slot_values(inst, slot):
    try:
        values = inst.getOwnSlotValues(slot) or []
    except Exception:
        values = []
    return [v for v in values if v is not None]

def frame_to_dict(inst, allowed=None, depth=0, max_depth=MAX_DEPTH_DEFAULT, visited=None):
    if visited is None:
        visited = set()

    base = get_id_name_class(inst)
    frame_id = base["id"]

    if frame_id in visited:
        return base
    if depth >= max_depth:
        return base

    visited.add(frame_id)
    data = {}

    try:
        slots = inst.getOwnSlots()
    except Exception:
        slots = []

    for slot in slots:
        if not slot:
            continue
        try:
            slot_name = str(slot.getName())
        except Exception:
            continue

        if slot_name.startswith(":"):
            continue
        if allowed is not None and slot_name not in allowed:
            continue

        values = iter_filled_slot_values(inst, slot)
        if not values:
            continue

        mapped = []
        for v in values:
            if is_frame(v):
                try:
                    mapped.append(
                        frame_to_dict(
                            v,
                            allowed=allowed,
                            depth=depth + 1,
                            max_depth=max_depth,
                            visited=visited,
                        )
                    )
                except Exception:
                    mapped.append(get_id_name_class(v))
            else:
                mapped.append(normalize_primitive(v))

        data[slot_name] = mapped[0] if len(mapped) == 1 else mapped

    return {**base, **data}

def filled_slots_dict(inst, allowed=None, max_depth=MAX_DEPTH_DEFAULT):
    d = frame_to_dict(inst, allowed=allowed, depth=0, max_depth=max_depth)
    return {k: v for k, v in d.items() if k not in ("id", "name", "className")}


def delete_instance(kb, inst) -> bool:
    """Best-effort deletion of a Protégé instance."""
    try:
        kb.deleteInstance(inst)
        return True
    except Exception:
        pass
    try:
        inst.delete()
        return True
    except Exception:
        return False


def rollback_instances(kb, instances: List[object]) -> None:
    """Remove previously created instances in reverse order."""
    for inst in reversed(instances):
        try:
            delete_instance(kb, inst)
        except Exception:
            continue


def persist_or_rollback(kb, created_frames: List[object]):
    """Persist the project, rolling back new instances on failure."""
    success, save_errors = save_project()
    if success:
        return True, None
    rollback_instances(kb, created_frames)
    return False, save_errors


def serialize_instance(instance, description: Optional[str], external_summary: Optional[dict]):
    """Serialize instance info for responses."""
    payload = frame_to_dict(instance, max_depth=2)
    if external_summary:
        payload["externalId"] = external_summary
    if description is not None and "description" not in payload:
        payload["description"] = description
    return payload


def create_instance_from_payload(kb, payload: dict, created_frames: List[object]):
    """Create an instance (and nested ones) from payload data."""
    if not isinstance(payload, dict):
        raise InstanceCreationError("Each instance payload must be an object.", 400)

    name = str(payload.get("name", "")).strip()
    class_name = str(payload.get("className", "")).strip()
    raw_description = payload.get("description")
    description = None if raw_description is None else str(raw_description)

    if not class_name:
        raise InstanceCreationError("Field 'className' is required.", 400)
    if not name:
        raise InstanceCreationError("Field 'name' is required.", 400)

    kb_cls = kb.getCls(class_name)
    if kb_cls is None:
        raise InstanceCreationError(f"Class '{class_name}' not found!", 404)

    if find_instance_by_slot_value(kb_cls, "name_", name):
        raise InstanceCreationError(f"Instance with name '{name}' already exists.", 409)

    external_payload = payload.get("externalId")
    try:
        external_id = validate_external_id(external_payload)
    except ValueError as exc:
        raise InstanceCreationError(str(exc), 400) from exc

    try:
        instance = kb.createInstance(None, kb_cls)
    except Exception as exc:
        raise InstanceCreationError(f"Failed to create instance: {exc}", 500) from exc

    created_frames.append(instance)

    if not set_instance_name(instance, name):
        raise InstanceCreationError("Failed to set instance name.", 500)
    if description is not None:
        if not set_slot_value(instance, "description", description):
            raise InstanceCreationError("Failed to set description.", 500)

    external_summary = None
    if external_id:
        repo_instance, repo_err = ensure_external_repository(kb, external_id["sourceName"])
        if repo_err:
            raise InstanceCreationError(repo_err, 500)

        _, ref_err = create_external_reference_record(
            kb,
            instance,
            repo_instance,
            external_id["id"],
            external_id["sourceName"],
        )
        if ref_err:
            raise InstanceCreationError(ref_err, 500)
        external_summary = external_id

    for slot_name, value in payload.items():
        if slot_name in RESERVED_FIELDS or value is None:
            continue

        slot = get_slot(kb, slot_name)
        if slot is None:
            raise InstanceCreationError(
                f"Slot '{slot_name}' not found for class '{class_name}'.",
                400,
            )

        values = value if isinstance(value, list) else [value]
        allows_multi = slot_allows_multiple(slot)
        if not allows_multi and len(values) > 1:
            raise InstanceCreationError(
                f"Slot '{slot_name}' does not allow multiple values.",
                400,
            )

        prepared_values = []
        for entry in values:
            if isinstance(entry, dict):
                nested = create_instance_from_payload(kb, entry, created_frames)
                prepared_values.append(nested.instance)
            else:
                prepared_values.append(entry)

        if allows_multi:
            for entry in prepared_values:
                if not add_slot_value(instance, slot, entry):
                    raise InstanceCreationError(
                        f"Failed to set slot '{slot_name}'.",
                        500,
                    )
        else:
            if not set_slot_value(instance, slot, prepared_values[0]):
                raise InstanceCreationError(
                    f"Failed to set slot '{slot_name}'.",
                    500,
                )

    return CreatedInstance(instance=instance, description=description, external_summary=external_summary)


@api_bp.post("/instances")
def create_instance():
    """Create a new instance in the Protégé knowledge base."""
    kb = get_knowledge_base()
    if kb is None:
        return jsonify({"error": "Knowledge Base not loaded!"}), 500

    payload, error = load_json_body()
    if payload is None:
        return jsonify({"error": error or "JSON body is required."}), 400

    created_frames: List[object] = []
    try:
        created = create_instance_from_payload(kb, payload, created_frames)
    except InstanceCreationError as exc:
        rollback_instances(kb, created_frames)
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as exc:  # pylint: disable=broad-except
        rollback_instances(kb, created_frames)
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    instance_payload = serialize_instance(
        created.instance,
        created.description,
        created.external_summary,
    )

    persisted, save_errors = persist_or_rollback(kb, created_frames)
    if not persisted:
        error_body = {"error": "Failed to persist Protégé project."}
        if save_errors:
            error_body["details"] = save_errors
        return jsonify(error_body), 500

    return (
        jsonify(
            {
                "message": "Instance created successfully.",
                "instance": instance_payload,
            }
        ),
        201,
    )


@api_bp.post("/instances/batch")
def create_instances_batch():
    """Create multiple instances (optionally with nested objects)."""
    kb = get_knowledge_base()
    if kb is None:
        return jsonify({"error": "Knowledge Base not loaded!"}), 500

    payload, error = load_json_body()
    if payload is None:
        return jsonify({"error": error or "JSON body is required."}), 400
    instances_payload = payload.get("instances")
    if not isinstance(instances_payload, list) or not instances_payload:
        return jsonify({"error": "Field 'instances' must be a non-empty array."}), 400

    created_frames: List[object] = []
    created_payloads = []
    try:
        for idx, instance_data in enumerate(instances_payload):
            try:
                created = create_instance_from_payload(kb, instance_data, created_frames)
            except InstanceCreationError as exc:
                raise InstanceCreationError(
                    f"Failed to create instance at index {idx}: {exc}",
                    exc.status_code,
                ) from exc

            created_payloads.append(
                serialize_instance(
                    created.instance,
                    created.description,
                    created.external_summary,
                )
            )
    except InstanceCreationError as exc:
        rollback_instances(kb, created_frames)
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as exc:  # pylint: disable=broad-except
        rollback_instances(kb, created_frames)
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    persisted, save_errors = persist_or_rollback(kb, created_frames)
    if not persisted:
        error_body = {"error": "Failed to persist Protégé project."}
        if save_errors:
            error_body["details"] = save_errors
        return jsonify(error_body), 500

    return (
        jsonify(
            {
                "message": f"{len(created_payloads)} instance(s) created successfully.",
                "instances": created_payloads,
            }
        ),
        201,
    )


@api_bp.get("/instances/<string:instance_id>")
def get_instance(instance_id: str):
    """Return details for a specific instance."""
    kb = get_knowledge_base()
    if kb is None:
        return jsonify({"error": "Knowledge Base not loaded!"}), 500

    inst = kb.getInstance(instance_id)
    if inst is None:
        return jsonify({"error": f"Instance '{instance_id}' not found."}), 404

    raw_slots = request.args.get("slots", "").strip()
    allowed = None
    if raw_slots:
        allowed_list = [s.strip() for s in raw_slots.split("^") if s.strip()]
        allowed = set(allowed_list) if allowed_list else None

    try:
        max_depth = int(request.args.get("maxdepth", MAX_DEPTH_DEFAULT))
    except Exception:
        max_depth = MAX_DEPTH_DEFAULT

    if max_depth < 1:
        max_depth = MAX_DEPTH_DEFAULT

    if max_depth > 3 and not allowed:
        return jsonify({
            "error": "Specific slots must be requested if the maximum depth is greater than 3.",
            "hint": "Use ?slots=slotA^slotB^slotC"
        }), 400

    data = frame_to_dict(inst, allowed=allowed, max_depth=max_depth)
    return jsonify({"instance": data, "maxDepthUsed": max_depth})


@api_bp.get("/classes/<string:class_name>/instances")
def list_instances_by_class(class_name):
    """
    Query params:
      - slots=slotA^slotB^slotC
      - maxdepth=N                 (se N>3 -> slots é obrigatório)
      - start=S                    (default 0; aceita 0-based ou 1-based)
      - count=C                    (default = todos a partir de start)
      - directinstances=true|false (default false -> inclui subclasses concretas)
    Retorno inclui: instances, count, total, nextPage (quando houver).
    """
    kb = get_knowledge_base()
    if kb is None:
        return jsonify({"error": "Knowledge Base not loaded!"}), 500

    cls = kb.getCls(class_name)
    if not cls:
        return jsonify({"error": f"Class '{class_name}' not found!"}), 404

    # --- slots ---
    raw_slots = request.args.get("slots", "").strip()
    allowed = None
    if raw_slots:
        allowed_list = [s.strip() for s in raw_slots.split("^") if s.strip()]
        allowed = set(allowed_list) if allowed_list else None

    # --- maxdepth ---
    try:
        max_depth = int(request.args.get("maxdepth", MAX_DEPTH_DEFAULT))
        if max_depth < 1:
            max_depth = MAX_DEPTH_DEFAULT
    except Exception:
        max_depth = MAX_DEPTH_DEFAULT

    if max_depth > 3 and not allowed:
        return jsonify({
            "error": "Specific slots must be requested if the maximum depth is greater than 3.",
            "hint": "Use ?slots=slotA^slotB^slotC"
        }), 400

    # --- directinstances ---
    raw_direct = request.args.get("directinstances", "false")
    direct_only = str(raw_direct).strip().lower() in ("1", "true", "t", "yes", "y", "on")

    # --- paginação: start / count ---
    # aceita start=0 (zero-based) ou 1-based; internamente converte para zero-based
    try:
        start_param = int(request.args.get("start", "0"))
    except Exception:
        start_param = 0
    zstart = 0 if start_param <= 0 else start_param - 1  # zero-based interno

    count_param = request.args.get("count", None)
    try:
        count_param = int(count_param) if count_param is not None else None
        if count_param is not None and count_param < 0:
            count_param = None
    except Exception:
        count_param = None

    # --- total e slicing ---
    # Pegamos a lista completa para paginar (ordem natural do Protégé)
    def is_abstract(c):
        try:
            return bool(c.isAbstract())
        except Exception:
            return False

    def unique_by_name(instances):
        seen = set()
        deduped = []
        for inst in instances:
            try:
                name = inst.getName()
            except Exception:
                name = None
            key = str(name) if name is not None else None
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(inst)
        return deduped

    def gather_instances(base_cls):
        if direct_only:
            try:
                return unique_by_name(list(base_cls.getDirectInstances() or []))
            except Exception:
                return []

        instances_collected = []
        try:
            subclasses = list(base_cls.getSubclasses() or [])
        except Exception:
            subclasses = []

        candidate_classes = [base_cls] + subclasses
        for c in candidate_classes:
            if is_abstract(c):
                continue
            try:
                insts = c.getInstances() or []
            except Exception:
                insts = []
            instances_collected.extend(insts)

        return unique_by_name(instances_collected)

    try:
        all_instances = gather_instances(cls)
    except Exception as e:
        return jsonify({"error": f"Failed to enumerate instances: {e}"}), 500

    total = len(all_instances)
    if zstart >= total:
        page_slice = []
    else:
        if count_param is None:
            zend = total
        else:
            zend = min(zstart + count_param, total)
        page_slice = all_instances[zstart:zend]

    returned_count = len(page_slice)

    # --- monta payload de cada instância ---
    def instance_payload(i):
        base = get_id_name_class(i)
        data = filled_slots_dict(i, allowed=allowed, max_depth=max_depth)
        if allowed is None or ("description" in allowed):
            data.setdefault("description", "")
        return {**base, **data}

    instances = [instance_payload(i) for i in page_slice]

    # --- nextPage ---
    next_page = None
    if returned_count > 0 and (zstart + returned_count) < total and count_param is not None:
        # Se o cliente usou start<=0 (zero-based), retornamos nextPage como 1-based (compat com seu exemplo)
        base_start_for_next = start_param if start_param > 0 else 1
        next_start = base_start_for_next + returned_count
        next_page = f"start={next_start},count={count_param}"

    response = {
        "class": class_name,
        "instances": instances,
        "count": returned_count,
        "total": total,
        "maxDepthUsed": max_depth,
    }
    if next_page:
        response["nextPage"] = next_page

    return jsonify(response)

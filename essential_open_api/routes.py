"""Blueprint with API endpoints for the Essential Open API."""

from flask import Blueprint, jsonify, request

from .jvm import (
    call_publish_async,
    get_knowledge_base,
    get_project,
    get_publish_status,
)

api_bp = Blueprint("api", __name__)

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

# ----- Essential Utility API -----

MAX_DEPTH_DEFAULT = 1

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

@api_bp.get("/classes/<string:class_name>/instances")
def list_instances_by_class(class_name):
    """
    Query params:
      - slots=slotA^slotB^slotC
      - maxdepth=N                 (se N>3 -> slots é obrigatório)
      - start=S                    (default 0; aceita 0-based ou 1-based)
      - count=C                    (default = todos a partir de start)
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
    try:
        all_instances = list(cls.getInstances())
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

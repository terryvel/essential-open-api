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

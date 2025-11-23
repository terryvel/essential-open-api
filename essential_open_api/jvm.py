"""Utilities to bootstrap the JVM required by the API."""

from __future__ import annotations

from typing import Optional, Tuple

import jpype

from .config import JARS_DIR, PPRJ_DIR, PPRJ_FILE

JARS_DIR.mkdir(parents=True, exist_ok=True)
PPRJ_DIR.mkdir(parents=True, exist_ok=True)

# Collect all jar files placed in the jars directory.
CLASSPATH = [str(jar.resolve()) for jar in JARS_DIR.glob("*.jar")]


def start_jvm() -> None:
    """Ensure the JVM is running with the configured classpath."""
    if jpype.isJVMStarted():
        return

    if not CLASSPATH:
        print(f"No .jar files found in {JARS_DIR}. Add them before starting the JVM.")
        return

    print("Starting JVM...")
    jpype.startJVM(classpath=CLASSPATH)
    print("JVM started successfully!")


_PROTEGE_PROJECT: Optional[object] = None
_KNOWLEDGE_BASE: Optional[object] = None


def load_pprj() -> Optional[Tuple[object, object]]:
    """Load the .pprj file and return the project and Knowledge Base."""
    try:
        global _PROTEGE_PROJECT, _KNOWLEDGE_BASE  # noqa: PLW0603

        if _KNOWLEDGE_BASE is not None:
            return _PROTEGE_PROJECT, _KNOWLEDGE_BASE

        start_jvm()

        if not PPRJ_FILE.exists():
            print(f"Error: The file {PPRJ_FILE} was not found!")
            return None

        protege_package = jpype.JPackage("edu.stanford.smi.protege.model")
        project_class = protege_package.Project

        print(f"Loading Protégé project: {PPRJ_FILE}...")
        project = project_class.loadProjectFromFile(str(PPRJ_FILE), [])

        kb = project.getKnowledgeBase()
        print("Knowledge Base loaded successfully!")
        _PROTEGE_PROJECT, _KNOWLEDGE_BASE = project, kb
        return project, kb

    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error while loading project: {exc}")
        return None


def get_knowledge_base() -> Optional[object]:
    """Return the loaded Knowledge Base, if available."""
    return _KNOWLEDGE_BASE


def get_project() -> Optional[object]:
    """Return the loaded Protégé project, if available."""
    return _PROTEGE_PROJECT


def call_publish_async(project: object, url: str, user: str, pwd: str) -> Optional[str]:
    """Trigger asynchronous publishing via PublishService."""
    try:
        start_jvm()
        publish_service = jpype.JClass("com.enterprise_architecture.essential_os.publish_service.PublishService")
        job_id = publish_service.startPublishAsync(project, url, user, pwd)
        return str(job_id)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error while starting publish job: {exc}")
        return None


def get_publish_status(job_id: str) -> Optional[Tuple[str, str]]:
    """Retrieve the status and logs of a publish job."""
    try:
        start_jvm()
        publish_service = jpype.JClass("com.enterprise_architecture.essential_os.publish_service.PublishService")
        status = publish_service.getPublishStatus(job_id)
        logs = publish_service.getPublishLogs(job_id)
        return str(status), str(logs)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error while checking publish status: {exc}")
        return None


def save_project():
    """Persist the current Protégé project to disk.

    Returns:
        tuple[bool, Optional[list[str]]]: (success flag, error messages if any).
    """
    project = get_project()
    if project is None:
        print("Cannot save project: Protégé project not loaded.")
        return False, ["Protégé project not loaded."]

    try:
        start_jvm()
        ArrayList = jpype.JClass("java.util.ArrayList")
        errors = ArrayList()

        save_method = getattr(project, "save", None)
        if callable(save_method):
            project.save(errors)
        else:
            legacy_save = getattr(project, "saveProject", None)
            if callable(legacy_save):
                legacy_save()
            else:
                print("Project object does not expose a save/saveProject method.")
                return False, ["Project object does not expose a save/saveProject method."]

        if hasattr(errors, "isEmpty") and not errors.isEmpty():
            error_messages = [str(err) for err in errors]
            print(f"Errors while saving project: {error_messages}")
            return False, error_messages

        print("Protégé project saved successfully.")
        return True, None
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error while saving project: {exc}")
        return False, [f"Error while saving project: {exc}"]

"""Factory function for the Essential Open API Flask application."""

from flask import Flask
from flask_cors import CORS

from .jvm import get_knowledge_base, load_pprj
from .routes import api_bp


def create_app() -> Flask:
    """Create and configure a Flask application instance."""
    app = Flask(__name__)

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    load_pprj()

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Simple health-check endpoint."""
        kb_loaded = get_knowledge_base() is not None
        return {"status": "ok", "kb_loaded": kb_loaded}

    return app


app = create_app()

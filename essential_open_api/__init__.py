"""Factory function for the Essential Open API Flask application."""

from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

from .jvm import get_knowledge_base, load_pprj
from .routes import api_bp


def create_app() -> Flask:
    """Create and configure a Flask application instance."""
    app = Flask(__name__)

    # Configure Flasgger
    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "Essential Open API",
            "description": "API for accessing the Protégé knowledge base.",
            "version": "1.0.0"
        },
        "basePath": "/"
    }
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/api/"
    }
    Swagger(app, template=swagger_template, config=swagger_config)

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    load_pprj()

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Simple health-check endpoint.
        ---
        tags:
          - System
        description: Returns the health status of the API and whether the knowledge base is loaded.
        responses:
          200:
            description: Health status object
            schema:
              type: object
              properties:
                status:
                  type: string
                  example: ok
                kb_loaded:
                  type: boolean
                  example: true
        """
        kb_loaded = get_knowledge_base() is not None
        return {"status": "ok", "kb_loaded": kb_loaded}

    return app


app = create_app()

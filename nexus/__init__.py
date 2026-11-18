"""NEXUS AI – Enterprise Intelligent Document Processing Platform."""
import logging

from flask import Flask

from .config import Config

__version__ = "1.0.0"


def create_app(config_overrides: dict | None = None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    for k, v in (config_overrides or {}).items():
        setattr(Config, k, v)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_MB * 1024 * 1024

    from .models import make_session_factory
    from .storage import get_storage

    app.extensions["nexus_session"] = make_session_factory(Config.DATABASE_URL)
    app.extensions["nexus_storage"] = get_storage()

    from .api import bp as api_bp
    from .web import bp as web_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(web_bp)

    # Warm up the classifier so the first request isn't slow.
    from .classifier import get_classifier

    get_classifier(Config.MODEL_DIR)
    return app

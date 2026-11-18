"""WSGI entry point: `gunicorn wsgi:app` or `python wsgi.py` for dev."""
import os

from nexus import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")

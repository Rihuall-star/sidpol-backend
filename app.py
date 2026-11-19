"""
Punto de entrada de SIDPOL (WSGI).

Render y Docker ejecutan `gunicorn app:app`. Toda la aplicación vive en el paquete `sidpol/`.
"""
import os

from sidpol import create_app

app = create_app()

if __name__ == "__main__":
    # El modo debug solo se activa explícitamente en local (nunca en producción)
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")

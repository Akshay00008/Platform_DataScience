from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
import os
from flask_cors import CORS
from utility.new_routes_index import routes

def create_app():
    
        # Initialize Flask app
    app = Flask(__name__)

    # Apply ProxyFix if you're behind a reverse proxy (e.g., Nginx, Apache)
    # Uncomment this if necessary based on your deployment setup
    # app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)

    # Enable CORS (Cross-Origin Resource Sharing)
    CORS(app)

    # Register routes blueprint
    app.register_blueprint(routes, url_prefix="/pythonservice")

    return app

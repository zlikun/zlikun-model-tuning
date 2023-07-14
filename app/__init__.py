from flask import Flask
from .routes import error, home, chatgpt


def create_flask_app():

    app = Flask(__name__)

    app.register_blueprint(error.router)
    app.register_blueprint(home.router)
    app.register_blueprint(chatgpt.router, url_prefix='/chatgpt/v1')

    return app

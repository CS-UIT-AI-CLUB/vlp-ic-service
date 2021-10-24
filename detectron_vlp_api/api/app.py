from flask import Flask
from flask_restful import Api
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload_images'
api = Api(app)

basedir = os.path.abspath(os.path.dirname(__file__))
os.makedirs('upload_images', exist_ok=True)
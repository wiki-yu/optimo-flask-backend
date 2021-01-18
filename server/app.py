from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
import base64

try:  # Python 3
    from urllib.parse import quote
except ImportError:  # Python 2
    from urllib import quote
from base64 import b64encode
from io import BytesIO


# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

app.config['UPLOAD_FOLDER'] = r"./upload"

@app.route('/motionDetection',methods=["POST"]) # 方法要与前端一致
def motionDetection():
    print("test!!!!!!!!!!!!!!!!!!!!!!")
    file_obj = request.files['file']  # Flask中获取文件
    if file_obj is None:
        return "No video file uploaded!!!!"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_obj.filename)     
    file_obj.save(file_path)

    f = open(file_path, 'rb')
    base64_str = base64.b64encode(f.read())
    # const detectedVideoUrl = `data:video/mp4;base64, ${base64_str}`
    # base64_str = "data:image/png;base64," + base64_str
    data_url = 'data:video/mp4;base64,{}'.format(quote(base64_str))
    return data_url

if __name__ == '__main__':
    app.run()
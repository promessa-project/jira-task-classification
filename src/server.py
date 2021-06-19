from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import models.predict_model as model
from paste.translogger import TransLogger
from waitress import serve

app = Flask(__name__, static_url_path='', static_folder='../web/static', template_folder='../web/templates')
app.config['JSON_SORT_KEYS'] = False # keep JSON in the order in which it is written

CORS(app)

@app.route('/docs', methods=['GET'])
def docs():
    return render_template('index.html')

@app.route('/predict-task', methods=['POST'])
def newproject():
    pred_result = model.predict_task(request.json)

    return jsonify(pred_result)

def run_server():
    global app

    port = 8090
    print('Listening on port:', port)

    serve(TransLogger(app), host='0.0.0.0', port=port)

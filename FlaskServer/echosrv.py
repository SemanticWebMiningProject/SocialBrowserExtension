from flask import Flask, Response, request
app = Flask(__name__)
@app.route('/<path:path>', methods=['GET', 'POST'])
@app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])

def catch_all(path):
        if request.data:
                print('Data:' + str(request.data))
        if request.form:
                print('Form:' + str(request.form))
        return ''

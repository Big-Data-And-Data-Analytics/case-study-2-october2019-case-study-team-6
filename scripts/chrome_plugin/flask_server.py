from flask import Flask
import flask


# initialization
app = Flask(__name__)
@app.route('/', methods=['POST'])
def yourMethod():
    response = flask.jsonify({'nat id': 'efficacy'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run()

from flask import Flask, render_template, request, redirect, url_for,make_response,json,jsonify
class Config(object):
    DEBUG=True
    JSON_AS_ASCII=False
app = Flask(__name__)
app.config.from_object(Config)
@app.route('/banner/images')
def banner_images():
    return json.dumps()


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
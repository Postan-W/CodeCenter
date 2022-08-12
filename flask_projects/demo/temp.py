from flask import Flask,request
class Config(object):
    DEBUG=True
    JSON_AS_ASCII=True
app = Flask(__name__)
@app.route("/getfile",methods=["GET","POST"])
def get_file():
    # form = request.form
    # file = form.get("file")
    return "收到"

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5001,debug=True)
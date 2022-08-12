"""
Date: 2022/03/15
"""
from flask import Flask, render_template, flash,request, redirect, url_for,make_response,json,jsonify,abort
from wtforms import StringField,PasswordField,SubmitField
from wtforms.validators import DataRequired,EqualTo
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_moment import Moment
app = Flask(__name__)
class Config(object):
    DEBUG = True
    JSON_AS_ASCII = False
    SECRET_KEY = "MINGZHU"
    #pip install pymysql，然后头写成如下形式。头写成mysql://的需要mysqlclient库(python2的叫mysqldb库)，但是容易安装失败
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:@127.0.0.1:3306/forflask"#冒号后面是密码，这里没有密码,最后是库名
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True#省略显式提交操作
app.config.from_object(Config)
bootstrap = Bootstrap(app)
moment = Moment(app)
class Register(FlaskForm):
    username = StringField(label="用户名",validators=[DataRequired("用户名不能为空")])
    password = PasswordField(label="密码",validators=[DataRequired("密码不能为空")])
    password2 = PasswordField(label="确认密码",validators=[DataRequired("密码不能为空"),EqualTo("password")])
    submit = SubmitField(label="提交")

@app.route("/flashdemo")
def flash_demo():
    form = Register()
    flash("这只是个测试")
    return render_template("bootstrap1.html",form=form)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)

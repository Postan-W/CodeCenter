from flask import Flask, render_template, request, redirect, url_for,make_response,json,jsonify
#jsonify的作用是将dict转为json样式的字符串，跟json.dumps(dict)作用差不多
class Config(object):
    DEBUG=True
    JSON_AS_ASCII=False
app = Flask(__name__)#这个name就是模块的名称__main__。flask用这个参数确定应用的位置，进而找到应用中其他文件的位置，例如图像和模板
app.config.from_object(Config)
# app.config['JSON_AS_ASCII'] = False,效果同上
#使用<>提取参数。函数中的形参名必须和<>中的一样，才能将这个值对应给函数作为实参
@app.route('/hello/<number>')#路由
def hello_world(number):#视图函数
    if int(number) == 1:
        return "Monday"
    elif int(number) == 2:
        return "Tuesday"
    else:
        return "Saturday"

#带转换器的int,float,string,path
@app.route('/number/<int:number>')#如果传的不能转为int就会报错
def get_number(number):
    if number == 1:
        return "Beijing"
    elif number == 2:
        return "Shenzhen"
    else:
        return "Shanghai"
#自定义转换器来匹配自己想要的参数形式
from werkzeug.routing import BaseConverter
class PhoneNumber(BaseConverter):
    def __init__(self,url_map,regex):
        super(PhoneNumber, self).__init__(url_map)
        self.regex = regex
    def to_python(self, value: str):
        print("重写父类方法")
        return value
#将转换器类加入到flask应用中
app.url_map.converters['self_build'] = PhoneNumber#之后用self_build这个key来找类
#定义一个使用转换器的接口
@app.route('/phonenumber/<self_build("1\d{10}"):value>',methods=["GET"])
def phonenumber(value):
    return "电话号码格式正确",600#状态码可以这样返回，还可以返回第三个内容，即首部，json格式

#---------------------返回页面------------------------------
@app.route("/login",methods=["GET","POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    elif request.method == "POST":
        name = request.form.get("name")
        password = request.form.get("password")
        if name == "百度":
            return redirect("https://www.baidu.com")#重定向,状态码为302
        else:
            return redirect(url_for("welcomefunc"))#重定向到其他接口。url_for有更丰富的参数形式，参见flask_web开发一书


@app.route("/welcome")
def welcomefunc():
    return "welcome"

#-------------------使用make_response返回数据---------------
"""如果像第一个接口直接返回字符串的话，flask会自动进行一些封装让他变成浏览器可以读取的格式，也就是content-type = text/html，状态码为200
使用make_response可以自定义自己的返回对象。
注：make_response返回网页也必须是render_template对象的形式
"""
@app.route("/customresponse/<int:number>")
def custom_response(number):
    if number == 1:
        # 定制header
        headers = {
            'content-type': 'text/plain'
        }
        #定制状态码(并且这里指定了返回的是普通文本，所以呈现的是<html>dsfasd</html>字符串，而不是dsfasd)
        response = make_response("<html>dsfasd</html>",1000)
        response.headers = headers
        return response
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
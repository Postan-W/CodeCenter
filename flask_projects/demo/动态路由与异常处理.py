"""
Date: 2022/03/12
"""
from flask import Flask, render_template, request, redirect, url_for,make_response,json,jsonify,abort
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
#在网页中抛出异常
@app.route("/number/<int:number>")
def get_number(number):
    if number == 1:
        return "first"
    elif number == 2:
        return "second"
    else:
        abort(404)#抛出异常

#自定义错误
@app.errorhandler(404)#处理abort的404异常
def handle_404(err):
    return render_template("error404.html")


#接口数据填充HTML
@app.route("/login2/<name>")
def login2_text(name):
    data2 = {"one":1,"two":[1,2,3,4,5]}#html中引用时dict也可以用点访问子元素(猜测背后还是转为了python支持的[]形式)
    data3 = '<h1>safe过滤器示例</h1>'
    return render_template("login2.html",data=name,data2 = data2,data3=data3)#这里的参数data(名字随便取)和html中的{{}}接收的保持一致

#Jinja2控制结构试验
@app.route("/control/<data>")
def control(data):
    data1 = ["FRUITS","Apple","Banana","Cherry"]
    data2 = ["CARS","MB","BMW","Ferrari"]
    if data == "data1":
        return render_template("jinja2控制结构.html",data1=data1,data2=data2)
    elif data == "data2":
        return render_template("macros.html",data1=data1,data2=data2)

@app.route("/indexpage")
def indexpage():
    return render_template("extends_base_index.html")

def number_filter(lis):
    return [i*2 for i in lis]
#注册过滤器，第一个参数是自定义的函数名，第二个是引用时的名称
app.add_template_filter(number_filter,"double")

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)

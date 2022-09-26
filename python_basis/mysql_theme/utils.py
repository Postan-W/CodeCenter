import pymysql

def get_connection(host="localhost",port=3306,user="root",passwd="",db="",charset="utf8"):
    return pymysql.connect(**vars())


def demo():
    connection = get_connection(db="forflask")
    cusor = connection.cursor()
    num = cusor.execute("select * from user")
    print("共查询到{}条".format(num))
    #一条一条取
    for i in range(num):
        print(cusor.fetchone())

    num = cusor.execute("select * from user")
    print(cusor.fetchall())
demo()



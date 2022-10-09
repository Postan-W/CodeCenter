#用a,b,c分别指第一、中间、第三根柱子。参数n表示第一根柱子上碟子的个数
def hano(a,b,c,n):
    if n == 1:
        print(a,"-->",c)
    else:
        hano(a,c,b,n-1)
        hano(a,b,c,1)
        hano(b,a,c,n-1)

hano('a','b','c',3)

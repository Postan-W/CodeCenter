import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def data_process():
    sales = pd.read_csv("./sales_data.csv")
    sales['ods_business_history.date_time'] = sales['ods_business_history.date_time'].apply(lambda d: time.strftime('%Y.%m.%d',time.strptime(d, '%Y/%m/%d')))
    print(sales.head(5))
    sales.to_csv("./salesdata.csv",index=0)
#绘制每连续七天的销售额
def graph_the_trend():
    sales = pd.read_csv("salesdata.csv")
    sales["ods_business_history.totalsales"] = sales["ods_business_history.totalsales"].apply(lambda x:int(x/10000000))
    plt.figure(figsize=(10, 4))
    plt.title('sales volume every 7 days')
    plt.xlabel('date point', fontsize=12)
    
    plt.ylabel('sales volume', fontsize=12)
    plt.plot(list(sales["ods_business_history.totalsales"]))
    plt.show()

# graph_the_trend()
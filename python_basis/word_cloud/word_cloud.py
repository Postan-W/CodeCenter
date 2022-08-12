"""
@Time : 2021/8/24 14:29
@Author : wmingzhu
@Annotation : 
"""
#stopwords:在英语里面会遇到很多a，the，or等使用频率很多的字或词，常为冠词、介词、副词或连词等。
#colormap，颜色映射
import matplotlib,jieba
from wordcloud import WordCloud,STOPWORDS

with open("article.txt","rb") as f:
    article = f.read()

#cut操作生成了一个包含分词结果的迭代器
words = jieba.cut(article)
# for i in words:
#     print(i)
#以空格分隔各个词形成一个字符串
words = " ".join(words)
#构造词云对象
words_cloud = WordCloud(background_color='white',width=500,height=300,max_font_size=80,min_font_size=20,mode='RGBA',font_path='STKAITI.TTF',stopwords=set(STOPWORDS))

words_cloud.generate(words)
words_cloud.to_file("wordscloud.png")
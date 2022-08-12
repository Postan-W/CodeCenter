"""
Date: 2022/04/19
"""
import urllib.request as ur
import urllib.parse as uparse
import json
#豆瓣电影，类型：动作
url = 'https://movie.douban.com/j/new_search_subjects?sort=U&range=0,10&tags=&start=0&genres=%E5%8A%A8%E4%BD%9C'
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'}
request = ur.Request(url=url,headers=header)
response = ur.urlopen(request)
content = response.read().decode('utf-8')
# content = json.loads(content)#将str类型loads成dict类型
print(content)
with open("./files/texts/action_films.json",'w',encoding='utf-8') as f:
    f.write(content)
"""
Date: 2022/04/18
"""
import urllib.request as ur
import urllib.parse as uparse
import json
#头中的cookie信息是最重要的，作为网站的反爬信息，其他的一般注释掉只保留cookie也能得到返回
header = {
'Accept': '*/*',
# 'Accept-Encoding': 'gzip, deflate, br',#该编码方式注释掉
'Accept-Language': 'zh-CN,zh;q=0.9',
'Connection': 'keep-alive',
'Content-Length': '132',
'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8,',
'Cookie': 'BIDUPSID=557719668236C7FC4468102DED13BEF2; PSTM=1619941829; __yjs_duid=1_69d334f7dd21a96ada2e43929e9ee7311620494728137; REALTIME_TRANS_SWITCH=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; SOUND_PREFER_SWITCH=1; SOUND_SPD_SWITCH=1; MCITY=-131%3A; APPGUIDE_10_0_2=1; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BDUSS=HMxci0wZThWQkxwNk9KS2hnNEp5cU5pT2RrUzBPSWlDUGhEY3JRVmwzb3Zlb0ppRVFBQUFBJCQAAAAAAAAAAAEAAACZ3Y6TsKHV0rXYt7241bvYvNIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC~tWmIv7Vpie; BDUSS_BFESS=HMxci0wZThWQkxwNk9KS2hnNEp5cU5pT2RrUzBPSWlDUGhEY3JRVmwzb3Zlb0ppRVFBQUFBJCQAAAAAAAAAAAEAAACZ3Y6TsKHV0rXYt7241bvYvNIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC~tWmIv7Vpie; BAIDUID=555148F34D42D1FCEB759A2E87F2D133:SL=0:NR=10:FG=1; BDRCVFR[w-kNo__JL0t]=1jmUUpB1KcCmh7GmLNEmi4WUvY; delPer=0; PSINO=2; H_PS_PSSID=36309_31660_34812_36165_34584_36121_35978_36280_36126_36226_26350_36314_36061; BA_HECTOR=04200ka10l8l000g9c1h5th600r; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1650115296,1650201460,1650293267,1650377920; Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1650377920; ab_sr=1.0.1_MzQ1YTEzMTE0NDdkMDVmYjY5YzAxNjg2MDhlNDg4MmEwNmJhMzgyMWMyMWU3M2JmNzY3OTU3YzlmYjVlN2RjMWM1NGU3YmU5NTViODBhNDE0ZDgzY2UzNGUzNTk3MDgzNTQyMGFjM2I3ODE5ZjI0MDE0ZDQ4NzI4Y2Y5ZTI2NTc2NzMwMWI0MGFjMTc1NGFjOGY4NGI0YWFmZDJhMDIzODFjZjcxY2RhM2JjZjczN2E3YjdiZmQ5MzBjNmNjOGMw',
'Host': 'fanyi.baidu.com',
'Origin': 'https://fanyi.baidu.com,',
'Referer': 'https://fanyi.baidu.com/?aldtype=16047',
'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
'sec-ch-ua-mobile': '?0',
'sec-ch-ua-platform': '"Windows"',
'Sec-Fetch-Dest': 'empty',
'Sec-Fetch-Mode': 'cors',
'Sec-Fetch-Site': 'same-origin',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36',
'X-Requested-With': 'XMLHttpRequest'
}
url = "https://fanyi.baidu.com/v2transapi?from=en&to=zh"
data = {
'from': 'en',
'to':'zh',
'query': 'love',
'transtype': 'enter',
'simple_means_flag': '3',
'sign': '198772.518981',
'token': '691f1eadb6a9755492e82e82a063ab63',
'domain': 'common',
}
#urlencode是将其转为from=en&to=zh&query=love这样的形式，但其仍为字符串格式，后面接个encode是将其转为byte
data = uparse.urlencode(data).encode('utf-8')
request = ur.Request(url=url,data=data,headers=header)
response = ur.urlopen(request)
content = response.read().decode('utf-8')
content = json.loads(content)
print(content)
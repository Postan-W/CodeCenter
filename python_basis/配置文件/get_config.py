from configparser import ConfigParser
#confpath = os.path.join(os.getcwd(), "config.ini")
conf = ConfigParser()
conf.read("/workspace/conf/config.ini", encoding="utf-8")
print(conf)
rtsp_list = conf["rtsp_url"].values()
camera_dict = {"camera0":{"url":"","table":"麦当劳-餐饮","shopid":"test20290","floor":"1","line":[(933,767),(1502, 977),(950,756),(1505,958)]},
               "camera1":{"url":"","table":"兰蔻-百货","shopid":"ACT20237","floor":"2","line":[(1059, 822),(1403,966),(1076,818),(1395,947)]},
               "camera2":{"url":"","table":"施华洛世奇-百货","shopid":"ACT20226","floor":"2","line":[(703,1045),(1827, 1419),(655,983), (1961, 1311)]},
               "camera3":{"url":"","table":"嘉禾一品-餐饮","shopid":"tets20291","floor":"2","line":[(1019,933),(1827,1181),(1050,917), (1831,1161)]},
               "camera4":{"url":"","table":"阿玛尼-百货","shopid":"ACT20232","floor":"2","line":[(1165,879), (1408,821),(1153,859),(1377, 814)]},
               "camera5":{"url":"","table":"麦当劳-百货","shopid":"test20290","floor":"1","line":[(311, 1205),(911, 981),(305,1185), (889,963)]},
               "camera6":{"url":"","table":"麦丝玛拉-百货","shopid":"ACT20225","floor":"2","line":[(1201, 591),(1433, 528),(1194,565), (1415, 515)]},
               "camera7":{"url":"","table":"潘多拉-百货","shopid":"ACT20236","floor":"2","line":[(1350, 819),(1609, 768),(1347,807), (1603, 757)]},
               "camera8":{"url":"","table":"MCM-百货","shopid":"ACT20227","floor":"2", "line":[(837,1075),(1241,1219),(877,1059), (1265, 1197)]},
               "camera9":{"url":"","table":"通道1-通道","shopid":"tongdao1","floor":"2","line":[(347,578),(1285,466),(349,597),(1285,481)]},
               "camera10":{"url":"","table":"通道2-通道","shopid":"tongdao2","floor":"2","line":[(311,525),(1469,619),(301,547),(1473,643)]}}

for i, rtsp_url in enumerate(rtsp_list):
    camera_id = "camera{0}".format(i)
    camera_dict[camera_id]["url"] = rtsp_url

cuda_id = conf.get("camera_info","cuda_id")
start_time = conf.get("camera_info","start_time")
stop_time = conf.getint("camera_info","stop_time")
db_provider = conf.get("db_info","provider")
db_host = conf.get("db_info","host")
db_port = conf.getint("db_info","port")
db_database = conf.get("db_info","database")
db_user = conf.get("db_info","user")
db_password = conf.get("db_info","password")




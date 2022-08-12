import xmltodict
import cv2
import multiprocessing
# from security import SecurityServer

str1 = '''<?xml version="1.0" encoding="UTF-8"?>\n<Notify><CmdType>VideoSecurityTaskAdd</CmdType><TaskID>46ea48dc93404317acebdcaf71c22dc9-0000</TaskID><StartDay>1656000000</StartDay><EndDay>1690473600</EndDay><AlgorithmList><AlgorithmConfig><StartTime>00:00:00</StartTime><EndTime>00:00:00</EndTime><EventType>73</EventType><EventName>\xe4\xba\xba\xe5\x91\x98\xe8\x81\x9a\xe9\x9b\x86</EventName><VideoWidth>1280</VideoWidth><VideoHeight>720</VideoHeight><MixDurationTime>0</MixDurationTime><Sensitivity></Sensitivity><Thtask_dictholdValue>3</Thtask_dictholdValue><DetectRegionList><DetectRegion><RuleID>2664-ea0b</RuleID><RuleName>2664-ea0b</RuleName><PointNum>4</PointNum><ShapeType>2</ShapeType><PointList><Point><PointX>6.643598615916955</PointX><PointY>26.57439446366782</PointY></Point><Point><PointX>6.643598615916955</PointX><PointY>695.363321799308</PointY></Point><Point><PointX>1253.4256055363321</PointX><PointY>695.363321799308</PointY></Point><Point><PointX>1253.4256055363321</PointX><PointY>26.57439446366782</PointY></Point></PointList></DetectRegion><DetectRegion><RuleID>2664-ea0b</RuleID><RuleName>2664-ea0b</RuleName><PointNum>4</PointNum><ShapeType>2</ShapeType><PointList><Point><PointX>6.643598615916955</PointX><PointY>26.57439446366782</PointY></Point><Point><PointX>6.643598615916955</PointX><PointY>695.363321799308</PointY></Point><Point><PointX>1253.4256055363321</PointX><PointY>695.363321799308</PointY></Point><Point><PointX>1253.4256055363321</PointX><PointY>26.57439446366782</PointY></Point></PointList></DetectRegion></DetectRegionList><DetectTripLineList/></AlgorithmConfig><AlgorithmConfig><StartTime>00:00:00</StartTime><EndTime>23:59:14</EndTime><EventType>25</EventType><EventName>\xe4\xba\xba\xe5\x91\x98\xe8\x84\xb1\xe5\xb2\x97</EventName><VideoWidth>1280</VideoWidth><VideoHeight>720</VideoHeight><MixDurationTime>60</MixDurationTime><Sensitivity>0</Sensitivity><Thtask_dictholdValue>2</Thtask_dictholdValue><DetectRegionList><DetectRegion><RuleID>45ff-fbd5</RuleID><RuleName>45ff-fbd5</RuleName><PointNum>4</PointNum><ShapeType>2</ShapeType><PointList><Point><PointX>376.47058823529414</PointX><PointY>239.16955017301038</PointY></Point><Point><PointX>376.47058823529414</PointX><PointY>655.5017301038063</PointY></Point><Point><PointX>852.5951557093425</PointX><PointY>655.5017301038063</PointY></Point><Point><PointX>852.5951557093425</PointX><PointY>239.16955017301038</PointY></Point></PointList></DetectRegion></DetectRegionList><DetectTripLineList/></AlgorithmConfig></AlgorithmList><ChannelInfo><ChannelID>46ea48dc93404317acebdcaf71c22dc9-0000</ChannelID><DeviceIP>172.16.67.250</DeviceIP><DevicePort>554</DevicePort><UserName>admin</UserName><PassWord>bonc123456</PassWord><DeviceType>1101</DeviceType><StreamNum>2</StreamNum><StreamServerIP>172.16.96.44</StreamServerIP><StreamServerPort>10088</StreamServerPort><PublicStreamServerIP>39.155.134.150</PublicStreamServerIP><PublicStreamServerPort>10088</PublicStreamServerPort><ChannelNum>0</ChannelNum><Info><ChannelIPAddtask_dicts>172.16.67.250</ChannelIPAddtask_dicts><ChannelPort>554</ChannelPort><ChannelPassword>bonc123456</ChannelPassword></Info></ChannelInfo></Notify>'''
task_dict = dict(xmltodict.parse(str1))
def processor(task_dict:dict,queue:multiprocessing.Queue=None)->None:
    def get_base_info(task_dict)->dict:
        info = {}
        info['TaskID'] = task_dict['Notify']['TaskID']
        info['ChannelID'] = task_dict['Notify']['ChannelInfo']['ChannelID']
        info['DeviceIP'] = task_dict['Notify']['ChannelInfo']['DeviceIP']
        info['DevicePort'] = task_dict['Notify']['ChannelInfo']['DevicePort']
        info['UserName'] = task_dict['Notify']['ChannelInfo']['UserName']
        info['PassWord'] = task_dict['Notify']['ChannelInfo']['PassWord']
        info['AlgorithmList'] = []
        if not type(task_dict['Notify']['AlgorithmList']['AlgorithmConfig']) == list:
            print("只有一个算法")
            algorithms = [task_dict['Notify']['AlgorithmList']['AlgorithmConfig']]
        else:
            print("有多个算法")
            algorithms = task_dict['Notify']['AlgorithmList']['AlgorithmConfig']

        for algorithm in algorithms:
            #默认每个算法的视频流的尺寸是一样的，所以后面的循环中的宽高覆盖前面的也没问题
            info['videoHeight'] = algorithm['VideoHeight']
            info['videoWidth'] = algorithm['VideoWidth']
            algorithm_info = {}
            algorithm_info['EventType'] = algorithm['EventType']
            algorithm_info['ThresholdValue'] = dict(algorithm).get('ThresholdValue', -100)
            if algorithm_info['ThresholdValue'] == -100:
                print("没有提供阈值")
            algorithm_info['region_list'] = []
            if not type(algorithm['DetectRegionList']['DetectRegion']) == list:
                print("只有一个区域")
                region_list = [algorithm['DetectRegionList']['DetectRegion']]
            else:
                print("有多个区域")
                region_list = algorithm['DetectRegionList']['DetectRegion']
            for region in region_list:
                region_points = []
                # 默认一个区域由多个点构成
                for point in region['PointList']['Point']:
                    region_points.append(list(point.values()))
                algorithm_info['region_list'].append(region_points)

            info['AlgorithmList'].append(algorithm_info)
        return info

    def get_connection_str(info):
        name = info['UserName']
        passwd = info['PassWord']
        port = info['DevicePort']
        ip = info['DeviceIP']
        uri = "rtsp://{}:{}@{}:{}/h264/ch1/main/av_stream".format(name, passwd, ip, port)
        normal_rtsp = uri
        gst_str = ""
        return (normal_rtsp,gst_str)

    def connect():
        uri = get_connection_str(get_base_info(task_dict))
        print(uri)
        try:
            capture = cv2.VideoCapture(uri[1], cv2.CAP_GSTREAMER)
            w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w == 0 and h == 0:
                raise Exception
            return capture
        except:
            print("调用gstreamer失败")
            try:
                capture = cv2.VideoCapture(uri[0])
                return capture
            except:
                print("直连rtsp失败")
                return None

    capture = connect()
    if not capture:
        print("没连上")
        error_info = None
        queue.put(error_info)
        return
    else:
        pass

info = {}
info['TaskID'] = task_dict['Notify']['TaskID']
info['ChannelID'] = task_dict['Notify']['ChannelInfo']['ChannelID']
info['DeviceIP'] = task_dict['Notify']['ChannelInfo']['DeviceIP']
info['DevicePort'] = task_dict['Notify']['ChannelInfo']['DevicePort']
info['UserName'] = task_dict['Notify']['ChannelInfo']['UserName']
info['PassWord'] = task_dict['Notify']['ChannelInfo']['PassWord']
info['endTime'] = int(task_dict['Notify']['EndDay'])
info['AlgorithmList'] = []
if not type(task_dict['Notify']['AlgorithmList']['AlgorithmConfig']) == list:
    print("只有一个算法")
    algorithms = [task_dict['Notify']['AlgorithmList']['AlgorithmConfig']]
else:
    print("有多个算法")
    algorithms = task_dict['Notify']['AlgorithmList']['AlgorithmConfig']

for algorithm in algorithms:
    # 默认每个算法的视频流的尺寸是一样的，所以后面的循环中的宽高覆盖前面的也没问题
    info['videoHeight'] = int(algorithm['VideoHeight'])
    info['videoWidth'] = int(algorithm['VideoWidth'])
    algorithm_info = {}
    algorithm_info['EventType'] = int(algorithm['EventType'])
    algorithm_info['ThresholdValue'] = int(dict(algorithm).get('ThresholdValue', -100))
    if algorithm_info['ThresholdValue'] == -100:
        print("没有提供阈值")
    algorithm_info['region_list'] = []
    if not type(algorithm['DetectRegionList']['DetectRegion']) == list:
        print("只有一个区域")
        region_list = [algorithm['DetectRegionList']['DetectRegion']]
    else:
        print("有多个区域")
        region_list = algorithm['DetectRegionList']['DetectRegion']
    for region in region_list:
        region_points = []
        # 默认一个区域由多个点构成
        for point in region['PointList']['Point']:
            region_points.append([float(item) for item in point.values()])
        algorithm_info['region_list'].append(region_points)

    info['AlgorithmList'].append(algorithm_info)
print(info)
for algo in info['AlgorithmList']:
    print(algo)

print(info['endTime'])
print(type(info['videoHeight']),info['videoHeight'],info['videoWidth'])
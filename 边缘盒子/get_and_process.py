import queue
import cv2
from security import SecurityServer
import threading
import time
"""
所有的业务均在函数processor中完成:解析任务信息、获取视频流、将视频流分配给各个模型处理、将处理完的结果放入调用者传入的result_queue中。
result_queue中放的信息是：
1.有返回结果时:(task_id,task_state=任务进行中,channel_id,event_type,image)，即(任务id,任务状态,摄像头id,事件id即对应的算法，处理结果)
2.连接视频流失败时：(task_id,task_state=流连接失败,channel_id,None,None)
3.任务时间到了：(task_id,task_state=任务完成,channel_id,None,None)
"""
def processor(task_dict:dict,result_queue:queue.Queue=None,security:SecurityServer=None)->None:
    '''
    :param task_dict: 该dict是由任务的xml直接解析得到的，即最外层是'Notify'这个key,请勿再次加工
    :param result_queue: 用来放置处理结果
    :param security:这个是封装模型的类对象
    :return:无返回值
    '''
    def get_base_info(task_dict)->dict:
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
            #默认每个算法的视频流的尺寸是一样的，所以后面的循环中的宽高覆盖前面的也没问题
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
        return info

    info = get_base_info(task_dict)

    def get_connection_str(info):
        name = info['UserName']
        passwd = info['PassWord']
        port = info['DevicePort']
        ip = info['DeviceIP']
        uri = "rtsp://{}:{}@{}:{}/h264/ch1/main/av_stream".format(name, passwd, ip, port)
        normal_rtsp = uri
        gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format("rtsp://{}:{}@{}:{}".format(name, passwd, ip, port),200,info['videoWidth'],info['videoHeight'])
        return (normal_rtsp,gst_str)

    def connect():
        uri = get_connection_str(info)
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
                print(uri[0])
                capture = cv2.VideoCapture(uri[0],cv2.CAP_FFMPEG)
                w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w == 0 and h == 0:
                    raise Exception
                return capture
            except:
                print("连接rtsp失败")
                return None

    def connect_cpu():
        uri = get_connection_str(info)
        capture = cv2.VideoCapture(uri[0], cv2.CAP_FFMPEG)
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print((w,h))
        if not (w == 0 and h ==0):
            print("连接成功")
            return capture
        else:
            return None

    capture = connect_cpu()
    def do_task(frame,event_type,threshhold,area,task_id,channel_id):
        if event_type == 72:
            alarm,image = security.electronic_fence(frame,area)
            if alarm:
                result = (task_id,"KWORKING",channel_id,event_type,image)
                result_queue.put(result)
        elif event_type == 73:
            alarm, image = security.person_gathered(frame,threshhold,area)
            if alarm:
                result = (task_id,"KWORKING",channel_id,event_type,image)
                result_queue.put(result)
        elif event_type == 25:
            alarm, image = security.staff_leave(frame,threshhold,area)
            if alarm:
                result = (task_id,"KWORKING",channel_id,event_type,image)
                result_queue.put(result)

    if not capture:
        print("连接有问题")
        error_info = (info['TaskID'],"KEYERROR",info['ChannelID'],None,None)
        result_queue.put(error_info)
        return
    else:
        while capture.isOpened():
            #采用多线程并行处理。后一个frame比前一个frame先处理完的情况也是可能发生的，需要商榷是否违背业务逻辑
            time.sleep(5)#延迟一秒读取，以免处理线程开辟太多硬件承受不了
            #任务到时判断
            if int(time.time()) > info['endTime']:
                print('任务到时')
                result_queue.put((info['TaskID'],"KEYDONE",info['ChannelID'],None,None))
                break

            ret, frame = capture.read()
            if not ret:
                error_info = (info['TaskID'], "KEYERROR", info['ChannelID'], None, None)
                result_queue.put(error_info)
                continue
            for algorithm in info['AlgorithmList']:
                t = threading.Thread(target=do_task,args=[frame,algorithm['EventType'],algorithm['ThresholdValue'],
                                                           algorithm['region_list'],info['TaskID'],info['ChannelID']])
                t.setDaemon(True)#主线程结束，该子线程结束
                t.start()









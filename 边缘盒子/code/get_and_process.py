import queue
import cv2
from security import SecurityServer
import threading
import time
import logging
import logging.config
logging.config.fileConfig('log.conf')
log = logging.getLogger("processor")

"""
所有的业务均在函数processor中完成:解析任务信息、获取视频流、将视频流分配给各个模型处理、将处理完的结果放入调用者传入的result_queue中。
result_queue中放的信息是：
1.有返回结果时:(task_id,task_state=任务进行中,channel_id,event_type,image)，即(任务id,任务状态,摄像头id,事件id即对应的算法，处理结果)
2.连接视频流失败时：(task_id,task_state=流连接失败,channel_id,None,None)
3.任务时间到了：(task_id,task_state=任务完成,channel_id,None,None)
"""

def do_task(task_dict:dict,result_queue:queue.Queue=None,security:SecurityServer=None)->None:
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
            log.info("只有一个算法要调用")
            algorithms = [task_dict['Notify']['AlgorithmList']['AlgorithmConfig']]
        else:
            log.info("有多个算法要调用")
            algorithms = task_dict['Notify']['AlgorithmList']['AlgorithmConfig']

        for algorithm in algorithms:
            #默认每个算法的视频流的尺寸是一样的，所以后面的循环中的宽高覆盖前面的也没问题
            info['videoHeight'] = int(algorithm['VideoHeight'])
            info['videoWidth'] = int(algorithm['VideoWidth'])
            algorithm_info = {}
            algorithm_info['EventType'] = int(algorithm['EventType'])
            algorithm_info['ThresholdValue'] = int(dict(algorithm).get('ThresholdValue', -100))
            if algorithm_info['ThresholdValue'] == -100:
                log.info("没有提供阈值")
            algorithm_info['region_list'] = []
            if not type(algorithm['DetectRegionList']['DetectRegion']) == list:
                region_list = [algorithm['DetectRegionList']['DetectRegion']]
            else:
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
        #一个pipeline的多个element之间通过 “!" 分隔，同时可以设置element及Cap的属性
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
            log.warning("调用gstreamer失败")
            try:
                print(uri[0])
                capture = cv2.VideoCapture(uri[0],cv2.CAP_FFMPEG)
                w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w == 0 and h == 0:
                    raise Exception
                return capture
            except:
                log.error("连接rtsp失败")
                return None

    def connect_cpu():
        uri = get_connection_str(info)
        capture = cv2.VideoCapture(uri[0], cv2.CAP_FFMPEG)
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print((w,h))
        if not (w == 0 and h ==0):
            log.info("rtsp连接成功")
            return capture
        else:
            return None

    capture = connect_cpu()

    def do_algorithm(algorithm_queue,event_type,threshhold,area,task_id,channel_id):
        if event_type == 72:
            while True:
                frame = algorithm_queue.get(block=1)
                alarm, image = security.electronic_fence(frame,area)
                if alarm:
                    result = (task_id,"KWORKING", channel_id, event_type, image)
                    result_queue.put(result)
                    log.info("任务{},事件{}的告警结果已放入队列".format(task_id, event_type))
        elif event_type == 73:
            while True:
                frame = algorithm_queue.get(block=1)
                alarm, image = security.person_gathered(frame, threshhold, area)
                if alarm:
                    result = (task_id, "KWORKING", channel_id, event_type, image)
                    result_queue.put(result)
                    log.info("任务{},事件{}的告警结果已放入队列".format(task_id, event_type))
        elif event_type == 25:
            while True:
                frame = algorithm_queue.get(block=1)
                alarm, image = security.staff_leave(frame, threshhold, area)
                if alarm:
                    result = (task_id, "KWORKING", channel_id, event_type, image)
                    result_queue.put(result)
                    log.info("任务{},事件{}的告警结果已放入队列".format(task_id, event_type))


    if not capture:
        log.error("连接失败")
        error_info = (info['TaskID'],"KEYERROR",info['ChannelID'],None,None)
        result_queue.put(error_info)
        log.info("任务{},摄像头{}的连接失败信息已放入队列".format(info['TaskID'],info['ChannelID']))
        return
    else:
        """
        本次任务用到几个算法就启用几个线程并行处理。主线程中为这n个算法创建n个queue，主线程中分别向n个queue中放frame,算法线程中使用frame
        """
        queue_list = []
        for algorithm in info['AlgorithmList']:
            queue_list.append(queue.Queue(100))
        log.info("创建了{}个处理队列".format(len(queue_list)))

        for algorithm in info['AlgorithmList']:
            t = threading.Thread(target=do_algorithm,args=[queue_list[info['AlgorithmList'].index(algorithm)], algorithm['EventType'], algorithm['ThresholdValue'],
                                                       algorithm['region_list'], info['TaskID'], info['ChannelID']])
            t.setDaemon(True)#主线程结束，该子线程结束
            t.start()
            log.info("算法线程{}开始运行".format(info['AlgorithmList'].index(algorithm)))


        #视频流连接可能没有中断，但一直读不到帧的情况也可能出现，所以设定一个阈值
        lose_frame = 0
        while capture.isOpened():
            time.sleep(3)#延迟读取，视硬件性能而定
            #任务到时判断
            if int(time.time()) > info['endTime']:
                log.warning('任务到时')
                result_queue.put((info['TaskID'],"KEYDONE",info['ChannelID'],None,None))
                log.info("任务{}到时信息已放入队列".format(info['TaskID']))
                break

            ret, frame = capture.read()
            if not ret:
                lose_frame += 1
                if lose_frame >= 50:
                    error_info = (info['TaskID'], "KEYERROR", info['ChannelID'], None, None)
                    result_queue.put(error_info)
                    log.error("视频读取失败，失败信息已放入队列")
                    break
                else:
                    continue
            for i in range(len(queue_list)):
                #queue_list[i].put(frame,block=1)。这里不能使用带阻塞的put方式，否则会影响到其他算法图像数据的存放
                if not queue_list[i].full():
                    queue_list[i].put(frame)

        #跳出while循环后意味着不再读视频流，无论是什么原因导致的，退出函数结束线程，被其守护的子线程也全部结束
        return


def processor(task_dict:dict,result_queue:queue.Queue=None,security:SecurityServer=None):
    t = threading.Thread(target=do_task,args=[task_dict,result_queue,security])
    t.setDaemon(True)
    t.start()
    return t








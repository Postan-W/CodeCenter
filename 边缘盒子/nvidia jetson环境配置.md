## 1.盒子基本信息

开发者可以利用强大的 NVIDIA Jetson AGX Xavier 开发套件大规模开发和部署智慧机器。它能够执行现代先进神经网络及其他 AI 工作负载，进而解决制造、物流、零售、服务、农业、智慧城市和便携式医疗器材的问题。NVIDIA嵌入式产品的核心模组不存在独立的显存和内存，CPU部分和GPU部分公用存储器的。所以Xavier 32GB LPDDR4x是CPU和GPU部分公用的，既是显存，也是内存。因为物理上的存储是统一的，连cudaMemcpy都可以完全省略，真正的0传输时间 . 举个例子说：Xavier在高速采集来自外设摄像头的信息，或者其他数据，采集速率假设达到了20GB/s，此时它们可以让GPU就地使用这些数据，完成计算。而可能台式GPU永远不可以，因为他们实际能取得的传输到自己显存的速度，由于必须经过PCI-E，只有10GB/s多点。此时将永远无法完成任务，卡在PCI-E传输瓶颈上。这种统一的CPU/GPU一体芯片，无此类问题。Xavier显存32GB，优势就更显著了：适合原本在大内存显卡上训练好的模型，直接挪动过来使用。可以同时容纳多种需要使用的网络（例如一个检测姿势的，一个检测汽车型号的），而无需反复的从内置EMMC存储载入。如果不介意训练时间稍微慢的话，则性价比突出，可以用来训练需要大量显存的模型，而无需购买动辄几万的台式显卡。注意：如果台式机想要拼凑32GB显存，只能双卡（至少RTX2080+的级别）+NVlink，成本几万。或者选择上大量的内存，然而大内存直接CPU训练，算力不够；而直接用GPU训练，PCI-E的带宽不够。Xavier是一个良好的综合，能用超过常见显卡的容量，无需拼凑，提供了良好的访存带宽+运算性能。另外不要忘记：Xavier的LPDDR4x内存（显存），理论速率有137GB/s，等于说，Xavier的GPU部分，可以用高达100多GB/s的速度，访问CPU部分产生的数据（例如CPU上的某些采集设备和它们的驱动，在内存中产生数据的速度）。 Jetson TX2是 58.3 GB/s，Jetson NANO更低，只有25.6 GB/s！

![image-20220713145743085](images/image-20220713145743085.png)

- NVIDIA Jetson AGX Xavier [16GB]
   * Jetpack 4.5.1 [L4T 32.5.1]
   * NV Power Mode: MODE_30W_ALL - Type: 3
   * jetson_stats.service: active
 - Libraries:
   * CUDA: 10.2.89
   * cuDNN: 8.0.0.180
   * TensorRT: 7.1.3.0
   * Visionworks: 1.6.0.501
   * OpenCV: 4.1.1 compiled CUDA: NO
   * VPI: ii libnvvpi1 1.0.15 arm64 NVIDIA Vision Programming Interface library
   * Vulkan: 1.2.70

## 2.常用操作命令

 - 查看动态配置与静态配置

   sudo jtop; sudo jtop -h

 - 查询系统相关信息

   jetson_release

## 3.Jetson AGX Xavier安装opencv4.5.4支持cuda和gstreamer

系统是 Jetpack 4.5.1，默认安装 opencv4.4.1，为了重新安装支持cuda的opencv，把当前的opencv卸载了,sudo apt purge libopencv*,然后再查看jetson_release：

![image-20220714101458831](images/image-20220714101458831.png)

重新安装：

参考链接1：https://blog.csdn.net/m0_62013374/article/details/125014338 (安装指导)

参考链接2：http://t.zoukankan.com/gloria-zhang-p-13819297.html  

参考链接3:https://blog.csdn.net/kittyzc/article/details/117388718（用来测试gstreamer是否正常）

参考链接4:https://blog.csdn.net/quicmous/article/details/124172950

- 下载源码

  opencv的下载地址：https://github.com/opencv/opencv/

  opencv_contrib的下载地址：https://github.com/opencv/opencv_contrib.git

  对opencv_contrib的解释：You can build OpenCV, so it will include the modules from this repository. Contrib modules are under constant development and it is recommended to use them alongside the master branch or latest releases of OpenCV.

  opencv和opencv_contrib选择一样的版本，复制code的zip包链接，

  在机器上wget下来解压。

  这里是将contrib文件夹移到了opencv里面(放哪都行，编译的时候指定文件夹位置即可)

- 安装依赖

  ```
  sudo add-apt-repository universe
  sudo add-apt-repository multiverse
  sudo apt-get update
  #包的安装顺序并非一定要像下面所示，下面只是一次成功配置环境的记录
  sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libavutil-dev libeigen3-dev libglew-dev libgtk2.0-dev libgtk-3-dev libpostproc-dev libswscale-dev libxvidcore-dev libx264-dev qt5-default zlib1g-dev libv4l-dev v4l-utils qv4l2 v4l2ucp
  注意eigen 库默认安装在了 /usr/include/eigen3/Eigen 路径下，但编译opencv的时候回去/usr/include/Eigen找库内容，找不到就报错了，所以设置个软连接：
  sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
  
  #gstreamer相关
  sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
  然后pkg-config --cflags --libs gstreamer-1.0
  
  #python环境相关
  sudo apt-get install -y python2.7-dev
  sudo apt-get install -y python3-dev python3-numpy python3-py python3-pytest
  sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libtiff5-dev libjasper-dev libdc1394-22-dev libavresample-dev
  其中可能会遇到依赖包libjasper-dev无法安装的问题:
  sudo add-apt-repository "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ xenial main multiverse restricted universe"
  sudo apt update
  sudo apt install libjasper1 libjasper-dev
  最后
  sudo apt-get update
  ```
  
  ![image-20220811114112850](images/image-20220811114112850.png)
  
- 编译

  在opencv源码文件夹内进行如下操作(本人阿里云盘中有opencv,opencv-contrib,.cache文件)：

  ```
  mkdir build
  cd build/
  cmake过程中会下载一些文件，因为网络问题可能会卡死，这些文件可以提前下载(一般在https://github.com/opencv/opencv_3rdparty里都能找到)，然后放到opencv源码目录下的.cache的对应目录中，这里编译的4.5.4版本的opencv，.cache目录直接在网上下载下来的(https://blog.csdn.net/length85/article/details/123128614)，替换源码包内的.cache目录。
  .cache中的文件是opencv_3rdparty中下载到的文件加上md5的前缀,所以如果是直接替换.cache目录，请注意文件名的问题。如果不直接替换则是修改opencv/3rdparty目录下的对应依赖工具的配置文件，比如这里ippicv，在opencv/3rdparty/ippicv下有ippicv.cmake文件，第47行是"https://raw.githubusercontent.com/opencv/opencv_3rdparty/${IPPICV_COMMIT}ippicv/"，即从这个链接下载ippicv_2019_lnx_intel64_general_20180723.tgz，然后改为从文件下载，file:该文件的所在目录，此时实际上就是从这个目录找文件，然后加了md5前缀，放到.cache下面去了。
  编译(各参数的含义参见文档最后):
  cmake \
  -D WITH_CUDA=ON
  -D CUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so
  -D CUDA_ARCH_BIN="7.2"  
  -D CUDA_ARCH_PTX=""
  -D BUILD_opencv_python3=ON
  -D BUILD_opencv_python2=ON 
  -D WITH_GSTREAMER=ON
  -D WITH_V4L=ON
  -D BUILD_TESTS=OFF 
  -D BUILD_PERF_TESTS=OFF  
  -D BUILD_EXAMPLES=OFF
  -D ENABLE_FAST_MATH=1
  -D CUDA_FAST_MATH=1 
  -D WITH_CUBLAS=1
  -D WITH_NVCUVID=ON
  -D CMAKE_BUILD_TYPE=RELEASE
  -D OPENCV_GENERATE_PKGCONFIG=1
  -D CMAKE_INSTALL_PREFIX=/usr/local
  -D OPENCV_EXTRA_MODULES_PATH=/home/nvidia/opencv-4.5.4/opencv_contrib-4.5.4/modules ..
  
  最后：
  #根据CPU核数进行编译
  make -j8
  #最后安装
  sudo make install
  ```

  

  ```
  删除opencv:
  1.到build的目录下执行sudo make unisntall(这里是build目录)
  2.删除安装目录下的关于opencv的内容(opencv的大小写可能不一样):
  sudo rm -rf /usr/local/share/opencv*
  sudo rm -rf /usr/local/include/opencv*
  sudo rm -rf /usr/local/lib/libopencv_*
  sudo rm -rf /usr/local/lib/cmake/opencv4
  3.把当前的源码目录也删了，以免和后来的混淆
  4.到安装目录下，这里是/usr，把相关的文件删干净
  find . -name "*opencv*" | xargs sudo rm -rf
  5.可以用Python导入cv2测试是否失败
  ```

  

- 配置opencv库

  sudo gedit /etc/ld.so.conf.d/opencv.conf ,末尾添加/usr/local/lib 

  sudo ldconfig 

  sudo gedit /etc/bash.bashrc

  末尾添加

  PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig  
  export PKG_CONFIG_PATH

  然后使生效

  source /etc/bash.bashrc

  sudo updatedb

- 测试

  pkg-config --modversion opencv

  python3 -c "import cv2; print(cv2.__version__)"

  使用cv2.getBuildInformation()查看OpenCV的构建信息

- 查看GPU

​		jetson_release

## 4.测试代码

- cpu(即opencv常规读流)

  ```
  import cv2
  import time
  #或者简写为：rtsp://admin:bonc123456@172.16.67.250
  url = "rtsp://admin:bonc123456@172.16.67.250:554/h264/ch1/main/av_stream"
  video_path = "./videos/testcpu.avi"
  def use_cpu(url,video_path):
      vid_cap = cv2.VideoCapture(url)
      w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = vid_cap.get(cv2.CAP_PROP_FPS)
      print((w, h), fps)
      fourcc = 'XVID'
      vid_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
  
      while vid_cap.isOpened():
          ret, frame = vid_cap.read()
          if not ret:
              print("连接错误")
              break
  
          vid_writer.write(frame)
  
      vid_cap.release()
  
  use_cpu(url=url,video_path=video_path)
  
  
  ```

- GPU

  

## 5.GStream介绍

Gstreamer是一个支持Windows，Linux，Android， iOS的跨平台的多媒体框架，应用程序可以通过管道（Pipeline）的方式，将多媒体处理的各个步骤串联起来，达到预期的效果。每个步骤通过元素（Element）基于GObject对象系统通过插件（plugins）的方式实现，方便了各项功能的扩展。

![img](images/c053d6e3181604e92f3c43f0a29589d0.png)

### Media Applications

最上面一层为应用，比如gstreamer自带的一些工具（gst-launch，gst-inspect等），以及基于gstreamer封装的库（gst-player，gst-rtsp-server，gst-editing-services等)根据不同场景实现的应用。

### Core Framework

中间一层为Core Framework，主要提供：

- 上层应用所需接口
- Plugin的框架
- Pipline的框架
- 数据在各个Element间的传输及处理机制
- 多个媒体流（Streaming）间的同步（比如音视频同步）
- 其他各种所需的工具库

### Plugins

最下层为各种插件，实现具体的数据处理及音视频输出，应用不需要关注插件的细节，会由Core Framework层负责插件的加载及管理。主要分类为：

- Protocols：负责各种协议的处理，file，http，rtsp等。
- Sources：负责数据源的处理，alsa，v4l2，tcp/udp等。
- Formats：负责媒体容器的处理，avi，mp4，ogg等。
- Codecs：负责媒体的编解码，mp3，vorbis等。
- Filters：负责媒体流的处理，converters，mixers，effects等。
- Sinks：负责媒体流输出到指定设备或目的地，alsa，xvideo，tcp/udp等。

Gstreamer框架根据各个模块的成熟度以及所使用的开源协议，将core及plugins置于不同的源码包中：

- gstreamer: 包含core framework及core elements。
- gst-plugins-base: gstreamer应用所需的必要插件。
- gst-plugins-good: 高质量的采用LGPL授权的插件。
- gst-plugins-ugly: 高质量，但使用了GPL等其他授权方式的库的插件，比如使用GPL的x264，x265。
- gst-plugins-bad: 质量有待提高的插件，成熟后可以移到good插件列表中。
- gst-libav: 对libav封装，使其能在gstreamer框架中使用。

### Gstreamer基础概念

#### Element

Element是Gstreamer中最重要的对象类型之一。一个element实现一个功能（读取文件，解码，输出等），程序需要创建多个element，并按顺序将其串连起来，构成一个完整的pipeline。

#### Pad

[src] ! [sink src] ! [sink src] ! [sink]，！分割了element,数据从!左边的element流向右边的element，当右边的element支持多种数据处理能力时，我们可以通过Cap来指定数据类型.
例如，下面的命令通过**Cap**指定了视频的宽高，videotestsrc会根据指定的宽高产生相应数据：

gst-launch-1.0 videotestsrc ! "video/x-raw,width=1280,height=720" ! autovideosink



element本身也可以指定属性，例如下文中的oggdemux name=demux，element和属性及属性之间用空格隔开。

Pad是一个element的输入/输出接口，分为src pad（生产数据）和sink pad（消费数据）两种。
两个element必须通过pad才能连接起来，pad拥有当前element能处理数据类型的能力（capabilities），会在连接时通过比较src pad和sink pad中所支持的能力，来选择最恰当的数据类型用于传输，如果element不支持，程序会直接退出。在element通过pad连接成功后，数据会从上一个element的src pad传到下一个element的sink pad然后进行处理。

#### Bin和Pipeline

Bin是一个容器，用于管理多个element，改变bin的状态时，bin会自动去修改所包含的element的状态，也会转发所收到的消息。如果没有bin，我们需要依次操作我们所使用的element。通过bin降低了应用的复杂度。
Pipeline继承自bin，为程序提供一个bus用于传输消息，并且对所有子element进行同步。当将pipeline的状态设置为PLAYING时，pipeline会在一个/多个新的线程中通过element处理数据。

下面我们通过一个文件播放的例子来熟悉上述提及的概念：测试文件[ sintel_trailer-480p.ogv](http://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.ogv)

gst-launch-1.0 filesrc location=sintel_trailer-480p.ogv ! oggdemux name=demux ! queue ! vorbisdec ! autoaudiosink demux. ! queue ! theoradec ! videoconvert ! autovideosink

通过上面的命令播放文件时，会创建如下pipeline：

![img](images/6819d7cd4606cf323acbcfb2199d5243.png)

可以看到这个pipeline由8个element构成，每个element都实现各自的功能：
filesrc读取文件，oggdemux解析文件，分别提取audio，video数据，queue缓存数据，vorbisdec解码audio，autoaudiosink自动选择音频设备并输出，theoradec解码video，videoconvert转换video数据格式，autovideosink自动选择显示设备并输出。

不同的element拥有不同数量及类型的pad，只有src pad的element被称为source element，只有sink pad的被称为sink element。

element可以同时拥有多个相同的pad，例如oggdemux在解析文件后，会将audio，video通过不同的pad输出。

### Gstreamer数据消息交互

在pipeline运行的过程中，各个element以及应用之间不可避免的需要进行数据消息的传输，gstreamer提供了bus系统以及多种数据类型（Buffers、Events、Messages，Queries）来达到此目的：

![img](images/7d1b1e2ef2b4381e842bc1bd5017997b.png)

#### Bus

Bus是gstreamer内部用于将消息从内部不同的streaming线程，传递到bus线程，再由bus所在线程将消息发送到应用程序。应用程序只需要向bus注册消息处理函数，即可接收到pipline中各element所发出的消息，使用bus后，应用程序就不用关心消息是从哪一个线程发出的，避免了处理多个线程同时发出消息的复杂性。

#### Buffers

用于从sources到sinks的媒体数据传输。

#### Events

用于element之间或者应用到element之间的信息传递，比如播放时的seek操作是通过event实现的。

#### Messages

是由element发出的消息，通过bus，以异步的方式被应用程序处理。通常用于传递errors, tags, state changes, buffering state, redirects等消息。消息处理是线程安全的。由于大部分消息是通过异步方式处理，所以会在应用程序里存在一点延迟，如果要及时的相应消息，需要在streaming线程捕获处理。

#### Queries

用于应用程序向gstreamer查询总时间，当前时间，文件大小等信息。

### gstreamer tools

Gstreamer自带了gst-inspect-1.0和gst-launch-1.0等其他命令行工具，我们可以使用这些工具完成常见的处理任务。
gst-inspect-1.0
查看gstreamer的plugin、element的信息。直接将plugin/element的类型作为参数，会列出其详细信息。如果不跟任何参数，会列出当前系统gstreamer所能查找到的所有插件。

$ gst-inspect-1.0 playbin

gst-launch-1.0
用于创建及执行一个Pipline，因此通常使用gst-launch先验证相关功能，然后再编写相应应用。
**通过上面ogg视频播放的例子，我们已经看到，一个pipeline的多个element之间通过 “!" 分隔，同时可以设置element及Cap的属性。例如：**
播放音视频：

gst-launch-1.0 playbin file:///home/root/test.mp4

转码：

gst-launch-1.0 filesrc location=/videos/sintel_trailer-480p.ogv ! decodebin name=decode ! \
               videoscale ! "video/x-raw,width=320,height=240" ! x264enc ! queue ! \
               mp4mux name=mux ! filesink location=320x240.mp4 decode. ! audioconvert ! \
               avenc_aac ! queue ! mux.

Streaming:

#Server
gst-launch-1.0 -v videotestsrc ! "video/x-raw,framerate=30/1" ! x264enc key-int-max=30 ! rtph264pay ! udpsink host=127.0.0.1 port=1234

#Client
gst-launch-1.0 udpsrc port=1234 ! "application/x-rtp, payload=96" ! rtph264depay ! decodebin ! autovideosink sync=false

### 使用Gstreamer可以调用rtsp流

```
import cv2
pipeline = "rtspsrc location=\"rtsp://admin:bonc123456@172.16.67.250:554/h264/ch1/main/av_stream\" ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=(string)BGRx! videoconvert ! appsink"
pipeline2 = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format("rtsp://{}:{}@{}:{}".format("admin","bonc123456","172.16.67.250",554),200,1280,720)

print(pipeline)
print(pipeline2)
capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

上面两种pipeline串的写法都可以

```

![image-20220721170820733](images/image-20220721170820733.png)

但是opecv的编译信息中并没有CUDA，所以后面需要将opencv(版本4.x)重新编译，支持CUDA与gstreamer

### cmake时的参数解释

```
//Path to a program.
ANT_EXECUTABLE:FILEPATH=D:/apache-ant-1.10.1/bin/ant.bat
 
//Build CUDA modules stubs when no CUDA SDK
BUILD_CUDA_STUBS:BOOL=OFF
 
//Create build rules for OpenCV Documentation
BUILD_DOCS:BOOL=ON
 
//Build all examples
BUILD_EXAMPLES:BOOL=OFF
 
//Build libjasper from source
BUILD_JASPER:BOOL=ON
 
//Build libjpeg from source
BUILD_JPEG:BOOL=ON
 
//Build openexr from source
BUILD_OPENEXR:BOOL=ON
 
//Enables 'make package_source' command
BUILD_PACKAGE:BOOL=ON
 
//Build performance tests
BUILD_PERF_TESTS:BOOL=ON
 
//Build libpng from source
BUILD_PNG:BOOL=ON
 
//Build shared libraries (.dll/.so) instead of static ones (.lib/.a)
BUILD_SHARED_LIBS:BOOL=ON
 
//Download and build TBB from source
BUILD_TBB:BOOL=OFF
 
//Build accuracy & regression tests
BUILD_TESTS:BOOL=ON
 
//Build libtiff from source
BUILD_TIFF:BOOL=ON
 
//Include debug info into debug libs (not MSCV only)
BUILD_WITH_DEBUG_INFO:BOOL=ON
 
//Enables dynamic linking of IPP (only for standalone IPP)
BUILD_WITH_DYNAMIC_IPP:BOOL=OFF
 
//Build zlib from source
BUILD_ZLIB:BOOL=ON
 
//Build utility applications (used for example to train classifiers)
BUILD_opencv_apps:BOOL=ON
 
//Include opencv_calib3d module into the OpenCV build
BUILD_opencv_calib3d:BOOL=ON
 
//Include opencv_core module into the OpenCV build
BUILD_opencv_core:BOOL=ON
 
//Include opencv_features2d module into the OpenCV build
BUILD_opencv_features2d:BOOL=ON
 
//Include opencv_flann module into the OpenCV build
BUILD_opencv_flann:BOOL=ON
 
//Include opencv_highgui module into the OpenCV build
BUILD_opencv_highgui:BOOL=ON
 
//Include opencv_imgcodecs module into the OpenCV build
BUILD_opencv_imgcodecs:BOOL=ON
 
//Include opencv_imgproc module into the OpenCV build
BUILD_opencv_imgproc:BOOL=ON
 
//Include opencv_java module into the OpenCV build
BUILD_opencv_java:BOOL=ON
 
//Include opencv_ml module into the OpenCV build
BUILD_opencv_ml:BOOL=ON
 
//Include opencv_objdetect module into the OpenCV build
BUILD_opencv_objdetect:BOOL=ON
 
//Include opencv_photo module into the OpenCV build
BUILD_opencv_photo:BOOL=ON
 
//Include opencv_shape module into the OpenCV build
BUILD_opencv_shape:BOOL=ON
 
//Include opencv_stitching module into the OpenCV build
BUILD_opencv_stitching:BOOL=ON
 
//Include opencv_superres module into the OpenCV build
BUILD_opencv_superres:BOOL=ON
 
//Include opencv_ts module into the OpenCV build
BUILD_opencv_ts:BOOL=ON
 
//Include opencv_video module into the OpenCV build
BUILD_opencv_video:BOOL=ON
 
//Include opencv_videoio module into the OpenCV build
BUILD_opencv_videoio:BOOL=ON
 
//Include opencv_videostab module into the OpenCV build
BUILD_opencv_videostab:BOOL=ON
 
//Include opencv_world module into the OpenCV build
BUILD_opencv_world:BOOL=OFF
 
//clAmdFft include directory
CLAMDBLAS_INCLUDE_DIR:PATH=CLAMDBLAS_INCLUDE_DIR-NOTFOUND
 
//AMD FFT root directory
CLAMDBLAS_ROOT_DIR:PATH=CLAMDBLAS_ROOT_DIR-NOTFOUND
 
//clAmdFft include directory
CLAMDFFT_INCLUDE_DIR:PATH=CLAMDFFT_INCLUDE_DIR-NOTFOUND
 
//AMD FFT root directory
CLAMDFFT_ROOT_DIR:PATH=CLAMDFFT_ROOT_DIR-NOTFOUND
 
//Path to a program.
CMAKE_AR:FILEPATH=C:/Qt/Tools/mingw530_32/bin/ar.exe
 
//Choose the type of build, options are: None(CMAKE_CXX_FLAGS or
// CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.
CMAKE_BUILD_TYPE:STRING=
 
//Enable/Disable color output during build.
CMAKE_COLOR_MAKEFILE:BOOL=ON
 
//Configs
CMAKE_CONFIGURATION_TYPES:STRING=Debug;Release
 
//CXX compiler
CMAKE_CXX_COMPILER:FILEPATH=C:/Qt/Tools/mingw530_32/bin/g++.exe
 
//Flags used by the compiler during all build types.
CMAKE_CXX_FLAGS:STRING=
 
//Flags used by the compiler during debug builds.
CMAKE_CXX_FLAGS_DEBUG:STRING=-g
 
//Flags used by the compiler during release builds for minimum
// size.
CMAKE_CXX_FLAGS_MINSIZEREL:STRING=-Os -DNDEBUG
 
//Flags used by the compiler during release builds.
CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG
 
//Flags used by the compiler during release builds with debug info.
CMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=-O2 -g -DNDEBUG
 
//Libraries linked by default with all C++ applications.
CMAKE_CXX_STANDARD_LIBRARIES:STRING=-lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32
 
//C compiler
CMAKE_C_COMPILER:FILEPATH=C:/Qt/Tools/mingw530_32/bin/gcc.exe
 
//Flags used by the compiler during all build types.
CMAKE_C_FLAGS:STRING=
 
//Flags used by the compiler during debug builds.
CMAKE_C_FLAGS_DEBUG:STRING=-g
 
//Flags used by the compiler during release builds for minimum
// size.
CMAKE_C_FLAGS_MINSIZEREL:STRING=-Os -DNDEBUG
 
//Flags used by the compiler during release builds.
CMAKE_C_FLAGS_RELEASE:STRING=-O3 -DNDEBUG
 
//Flags used by the compiler during release builds with debug info.
CMAKE_C_FLAGS_RELWITHDEBINFO:STRING=-O2 -g -DNDEBUG
 
//Libraries linked by default with all C applications.
CMAKE_C_STANDARD_LIBRARIES:STRING=-lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32
 
//Flags used by the linker.
CMAKE_EXE_LINKER_FLAGS:STRING=
 
//Flags used by the linker during debug builds.
CMAKE_EXE_LINKER_FLAGS_DEBUG:STRING=
 
//Flags used by the linker during release minsize builds.
CMAKE_EXE_LINKER_FLAGS_MINSIZEREL:STRING=
 
//Flags used by the linker during release builds.
CMAKE_EXE_LINKER_FLAGS_RELEASE:STRING=
 
//Flags used by the linker during Release with Debug Info builds.
CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO:STRING=
 
//Convert GNU import libraries to MS format (requires Visual Studio)
CMAKE_GNUtoMS:BOOL=OFF
 
//Installation Directory
CMAKE_INSTALL_PREFIX:PATH=F:/opencv/sources/opencv3/install
 
//Path to a program.
CMAKE_LINKER:FILEPATH=C:/Qt/Tools/mingw530_32/bin/ld.exe
 
//Path to a program.
CMAKE_MAKE_PROGRAM:FILEPATH=C:/Qt/Tools/mingw530_32/bin/mingw32-make.exe
 
//Flags used by the linker during the creation of modules.
CMAKE_MODULE_LINKER_FLAGS:STRING=
 
//Flags used by the linker during debug builds.
CMAKE_MODULE_LINKER_FLAGS_DEBUG:STRING=
 
//Flags used by the linker during release minsize builds.
CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL:STRING=
 
//Flags used by the linker during release builds.
CMAKE_MODULE_LINKER_FLAGS_RELEASE:STRING=
 
//Flags used by the linker during Release with Debug Info builds.
CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO:STRING=
 
//Path to a program.
CMAKE_NM:FILEPATH=C:/Qt/Tools/mingw530_32/bin/nm.exe
 
//Path to a program.
CMAKE_OBJCOPY:FILEPATH=C:/Qt/Tools/mingw530_32/bin/objcopy.exe
 
//Path to a program.
CMAKE_OBJDUMP:FILEPATH=C:/Qt/Tools/mingw530_32/bin/objdump.exe
 
//Value Computed by CMake
CMAKE_PROJECT_NAME:STATIC=OpenCV
 
//Path to a program.
CMAKE_RANLIB:FILEPATH=C:/Qt/Tools/mingw530_32/bin/ranlib.exe
 
//RC compiler
CMAKE_RC_COMPILER:FILEPATH=C:/Qt/Tools/mingw530_32/bin/windres.exe
 
//Flags for Windows Resource Compiler.
CMAKE_RC_FLAGS:STRING=
 
//Path to a program.
CMAKE_SH:FILEPATH=CMAKE_SH-NOTFOUND
 
//Flags used by the linker during the creation of dll's.
CMAKE_SHARED_LINKER_FLAGS:STRING=
 
//Flags used by the linker during debug builds.
CMAKE_SHARED_LINKER_FLAGS_DEBUG:STRING=
 
//Flags used by the linker during release minsize builds.
CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL:STRING=
 
//Flags used by the linker during release builds.
CMAKE_SHARED_LINKER_FLAGS_RELEASE:STRING=
 
//Flags used by the linker during Release with Debug Info builds.
CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO:STRING=
 
//If set, runtime paths are not added when installing shared libraries,
// but are added when building.
CMAKE_SKIP_INSTALL_RPATH:BOOL=NO
 
//If set, runtime paths are not added when using shared libraries.
CMAKE_SKIP_RPATH:BOOL=NO
 
//Flags used by the linker during the creation of static libraries.
CMAKE_STATIC_LINKER_FLAGS:STRING=
 
//Flags used by the linker during debug builds.
CMAKE_STATIC_LINKER_FLAGS_DEBUG:STRING=
 
//Flags used by the linker during release minsize builds.
CMAKE_STATIC_LINKER_FLAGS_MINSIZEREL:STRING=
 
//Flags used by the linker during release builds.
CMAKE_STATIC_LINKER_FLAGS_RELEASE:STRING=
 
//Flags used by the linker during Release with Debug Info builds.
CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO:STRING=
 
//Path to a program.
CMAKE_STRIP:FILEPATH=C:/Qt/Tools/mingw530_32/bin/strip.exe
 
//If this value is on, makefiles will be generated without the
// .SILENT directive, and all commands will be echoed to the console
// during the make.  This is useful for debugging only. With Visual
// Studio IDE projects all commands are done without /nologo.
CMAKE_VERBOSE_MAKEFILE:BOOL=FALSE
 
//The path to cmu1394 headers
CMU1394_INCLUDE_PATH:PATH=CMU1394_INCLUDE_PATH-NOTFOUND
 
//Enable to build 7-Zip packages
CPACK_BINARY_7Z:BOOL=OFF
 
//Enable to build IFW packages
CPACK_BINARY_IFW:BOOL=OFF
 
//Enable to build NSIS packages
CPACK_BINARY_NSIS:BOOL=ON
 
//Enable to build WiX packages
CPACK_BINARY_WIX:BOOL=OFF
 
//Enable to build ZIP packages
CPACK_BINARY_ZIP:BOOL=OFF
 
//Enable to build 7-Zip source packages
CPACK_SOURCE_7Z:BOOL=ON
 
//Enable to build ZIP source packages
CPACK_SOURCE_ZIP:BOOL=ON
 
//The path to DC1394 2.x headers
DC1394_2_INCLUDE_PATH:PATH=DC1394_2_INCLUDE_PATH-NOTFOUND
 
//Download external test data (Python executable and OPENCV_TEST_DATA_PATH
// environment variable may be required)
DOWNLOAD_EXTERNAL_TEST_DATA:BOOL=OFF
 
//Graphviz Dot tool for using Doxygen
DOXYGEN_DOT_EXECUTABLE:FILEPATH=DOXYGEN_DOT_EXECUTABLE-NOTFOUND
 
//Doxygen documentation generation tool (http://www.doxygen.org)
DOXYGEN_EXECUTABLE:FILEPATH=DOXYGEN_EXECUTABLE-NOTFOUND
 
//The path to Eigen3/Eigen2 headers
EIGEN_INCLUDE_PATH:PATH=EIGEN_INCLUDE_PATH-NOTFOUND
 
//Enable AVX instructions
ENABLE_AVX:BOOL=OFF
 
//Enable AVX2 instructions
ENABLE_AVX2:BOOL=OFF
 
//Use ccache
ENABLE_CCACHE:BOOL=OFF
 
//Enable coverage collection with  GCov
ENABLE_COVERAGE:BOOL=OFF
 
//Enable -ffast-math (not recommended for GCC 4.6.x)
ENABLE_FAST_MATH:BOOL=OFF
 
//Enable FMA3 instructions
ENABLE_FMA3:BOOL=OFF
 
//Collect implementation data on function call
ENABLE_IMPL_COLLECTION:BOOL=OFF
 
//Instrument functions to collect calls trace and performance
ENABLE_INSTRUMENTATION:BOOL=OFF
 
//Show all warnings even if they are too noisy
ENABLE_NOISY_WARNINGS:BOOL=OFF
 
//Enable -fomit-frame-pointer for GCC
ENABLE_OMIT_FRAME_POINTER:BOOL=ON
 
//Enable POPCNT instructions
ENABLE_POPCNT:BOOL=OFF
 
//Use precompiled headers
ENABLE_PRECOMPILED_HEADERS:BOOL=ON
 
//Enable profiling in the GCC compiler (Add flags: -g -pg)
ENABLE_PROFILING:BOOL=OFF
 
//Solution folder in Visual Studio or in other IDEs
ENABLE_SOLUTION_FOLDERS:BOOL=OFF
 
//Enable SSE instructions
ENABLE_SSE:BOOL=ON
 
//Enable SSE2 instructions
ENABLE_SSE2:BOOL=ON
 
//Enable SSE3 instructions
ENABLE_SSE3:BOOL=ON
 
//Enable SSE4.1 instructions
ENABLE_SSE41:BOOL=OFF
 
//Enable SSE4.2 instructions
ENABLE_SSE42:BOOL=OFF
 
//Enable SSSE3 instructions
ENABLE_SSSE3:BOOL=OFF
 
//Output directory for applications
EXECUTABLE_OUTPUT_PATH:PATH=F:/opencv/sources/opencv3/bin
 
//Git command line client
GIT_EXECUTABLE:FILEPATH=C:/Program Files/Git/cmd/git.exe
 
//Glib library
GLIB_LIBRARY:FILEPATH=GLIB_LIBRARY-NOTFOUND
 
//gstcdda library to link with
GLIB_gstcdda_LIBRARY:FILEPATH=GLIB_gstcdda_LIBRARY-NOTFOUND
 
//Glib library
GOBJECT_LIBRARY:FILEPATH=GOBJECT_LIBRARY-NOTFOUND
 
//Directory containing glib.h include file
GSTREAMER_glib_INCLUDE_DIR:PATH=GSTREAMER_glib_INCLUDE_DIR-NOTFOUND
 
//Directory containing glibconfig.h include file
GSTREAMER_glibconfig_INCLUDE_DIR:PATH=GSTREAMER_glibconfig_INCLUDE_DIR-NOTFOUND
 
//Directory containing gst/gst.h include file
GSTREAMER_gst_INCLUDE_DIR:PATH=GSTREAMER_gst_INCLUDE_DIR-NOTFOUND
 
//gstapp library to link with
GSTREAMER_gstapp_LIBRARY:FILEPATH=GSTREAMER_gstapp_LIBRARY-NOTFOUND
 
//gstaudio library to link with
GSTREAMER_gstaudio_LIBRARY:FILEPATH=GSTREAMER_gstaudio_LIBRARY-NOTFOUND
 
//gstbase library to link with
GSTREAMER_gstbase_LIBRARY:FILEPATH=GSTREAMER_gstbase_LIBRARY-NOTFOUND
 
//Directory containing gst/gstconfig.h include file
GSTREAMER_gstconfig_INCLUDE_DIR:PATH=GSTREAMER_gstconfig_INCLUDE_DIR-NOTFOUND
 
//gstcontroller library to link with
GSTREAMER_gstcontroller_LIBRARY:FILEPATH=GSTREAMER_gstcontroller_LIBRARY-NOTFOUND
 
//gstnet library to link with
GSTREAMER_gstnet_LIBRARY:FILEPATH=GSTREAMER_gstnet_LIBRARY-NOTFOUND
 
//gstpbutils library to link with
GSTREAMER_gstpbutils_LIBRARY:FILEPATH=GSTREAMER_gstpbutils_LIBRARY-NOTFOUND
 
//gstreamer library to link with
GSTREAMER_gstreamer_LIBRARY:FILEPATH=GSTREAMER_gstreamer_LIBRARY-NOTFOUND
 
//gstriff library to link with
GSTREAMER_gstriff_LIBRARY:FILEPATH=GSTREAMER_gstriff_LIBRARY-NOTFOUND
 
//gstrtp library to link with
GSTREAMER_gstrtp_LIBRARY:FILEPATH=GSTREAMER_gstrtp_LIBRARY-NOTFOUND
 
//gstrtsp library to link with
GSTREAMER_gstrtsp_LIBRARY:FILEPATH=GSTREAMER_gstrtsp_LIBRARY-NOTFOUND
 
//gstsdp library to link with
GSTREAMER_gstsdp_LIBRARY:FILEPATH=GSTREAMER_gstsdp_LIBRARY-NOTFOUND
 
//gsttag library to link with
GSTREAMER_gsttag_LIBRARY:FILEPATH=GSTREAMER_gsttag_LIBRARY-NOTFOUND
 
//gstvideo library to link with
GSTREAMER_gstvideo_LIBRARY:FILEPATH=GSTREAMER_gstvideo_LIBRARY-NOTFOUND
 
//True if MKL found
HAVE_MKL:BOOL=OFF
 
//Change install rules to build the distribution package
INSTALL_CREATE_DISTRIB:BOOL=OFF
 
//Install C examples
INSTALL_C_EXAMPLES:BOOL=OFF
 
//Install Python examples
INSTALL_PYTHON_EXAMPLES:BOOL=OFF
 
//Install accuracy and performance test binaries and test data
INSTALL_TESTS:BOOL=OFF
 
//Dependencies for the target
IlmImf_LIB_DEPENDS:STATIC=general;zlib;
 
//Path to a file.
JAVA_AWT_INCLUDE_PATH:PATH=C:/Program Files/Java/jdk1.8.0_45/include
 
//Path to a library.
JAVA_AWT_LIBRARY:FILEPATH=C:/Program Files/Java/jdk1.8.0_45/lib/jawt.lib
 
//Path to a file.
JAVA_INCLUDE_PATH:PATH=C:/Program Files/Java/jdk1.8.0_45/include
 
//Path to a file.
JAVA_INCLUDE_PATH2:PATH=C:/Program Files/Java/jdk1.8.0_45/include/win32
 
//Path to a library.
JAVA_JVM_LIBRARY:FILEPATH=C:/Program Files/Java/jdk1.8.0_45/lib/jvm.lib
 
//Alternative name of cblas.h
LAPACK_CBLAS_H:STRING=
 
//Lapack implementation id
LAPACK_IMPL:STRING=Unknown
 
//Path to BLAS include dir
LAPACK_INCLUDE_DIR:PATH=
 
//Alternative name of lapacke.h
LAPACK_LAPACKE_H:STRING=
 
//Names of BLAS & LAPACK binaries (.so, .dll, .a, .lib)
LAPACK_LIBRARIES:STRING=
 
//Path to a program.
MATLAB_BIN_:FILEPATH=D:/Program Files/MATLAB/R2016b/bin/matlab.exe
 
//Path to a file.
MATLAB_INCLUDE_DIRS_:PATH=D:/Program Files/MATLAB/R2016b/extern/include
 
//Path to a library.
MATLAB_LIB_MAT_:FILEPATH=D:/Program Files/MATLAB/R2016b/bin/win64/libmat.dll
 
//Path to a library.
MATLAB_LIB_MEX_:FILEPATH=D:/Program Files/MATLAB/R2016b/bin/win64/libmex.dll
 
//Path to a library.
MATLAB_LIB_MX_:FILEPATH=D:/Program Files/MATLAB/R2016b/bin/win64/libmx.dll
 
//Path to a file.
MATLAB_MEXEXT_SCRIPT_:FILEPATH=D:/Program Files/MATLAB/R2016b/bin/mexext.bat
 
//Path to a program.
MATLAB_MEX_SCRIPT_:FILEPATH=D:/Program Files/MATLAB/R2016b/bin/mex.bat
 
//Path to a program.
MATLAB_ROOT_DIR_:FILEPATH=D:/Program Files/MATLAB/R2016b/bin/matlab.exe
 
//Path to a file.
MKL_ROOT_DIR:PATH=MKL_ROOT_DIR-NOTFOUND
 
//Use MKL with OpenMP multithreading
MKL_WITH_OPENMP:BOOL=OFF
 
//Use MKL with TBB multithreading
MKL_WITH_TBB:BOOL=OFF
 
//OpenCL library is found
OPENCL_FOUND:BOOL=ON
 
//OpenCL include directory
OPENCL_INCLUDE_DIR:PATH=F:/opencv/sources/3rdparty/include/opencl/1.2
 
//OpenCL library
OPENCL_LIBRARY:STRING=
 
//Where to create the platform-dependant cvconfig.h
OPENCV_CONFIG_FILE_INCLUDE_DIR:PATH=F:/opencv/sources/opencv3
 
//Enable non-free algorithms
OPENCV_ENABLE_NONFREE:BOOL=OFF
 
//Where to look for additional OpenCV modules
OPENCV_EXTRA_MODULES_PATH:PATH=
 
//Treat warnings as errors
OPENCV_WARNINGS_ARE_ERRORS:BOOL=OFF
 
//Path to a file.
OpenBLAS_INCLUDE_DIR:PATH=OpenBLAS_INCLUDE_DIR-NOTFOUND
 
//Path to a library.
OpenBLAS_LIB:FILEPATH=OpenBLAS_LIB-NOTFOUND
 
//Value Computed by CMake
OpenCV_BINARY_DIR:STATIC=F:/opencv/sources/opencv3
 
//The directory containing a CMake configuration file for OpenCV_HAL.
OpenCV_HAL_DIR:PATH=OpenCV_HAL_DIR-NOTFOUND
 
//Value Computed by CMake
OpenCV_SOURCE_DIR:STATIC=F:/opencv/sources
 
//Path to Python interpretor
PYTHON2_EXECUTABLE:FILEPATH=C:/yotta/python/python.exe
 
//Python include dir
PYTHON2_INCLUDE_DIR:PATH=C:/yotta/python/include
 
//Python include dir 2
PYTHON2_INCLUDE_DIR2:PATH=
 
//Path to Python library
PYTHON2_LIBRARY:FILEPATH=C:/yotta/python/libs/libpython27.a
 
//Path to Python debug
PYTHON2_LIBRARY_DEBUG:FILEPATH=PYTHON_DEBUG_LIBRARY-NOTFOUND
 
//Path to numpy headers
PYTHON2_NUMPY_INCLUDE_DIRS:PATH=
 
//Where to install the python packages.
PYTHON2_PACKAGES_PATH:PATH=C:/yotta/python/Lib/site-packages
 
//Path to Python interpretor
PYTHON3_EXECUTABLE:FILEPATH=
 
//Python include dir
PYTHON3_INCLUDE_DIR:PATH=
 
//Python include dir 2
PYTHON3_INCLUDE_DIR2:PATH=
 
//Path to Python library
PYTHON3_LIBRARY:FILEPATH=
 
//Path to Python debug
PYTHON3_LIBRARY_DEBUG:FILEPATH=
 
//Path to numpy headers
PYTHON3_NUMPY_INCLUDE_DIRS:PATH=
 
//Where to install the python packages.
PYTHON3_PACKAGES_PATH:PATH=
 
//Path to a program.
PYTHON_EXECUTABLE:FILEPATH=C:/yotta/python/python.exe
 
//The directory containing a CMake configuration file for VTK.
VTK_DIR:PATH=VTK_DIR-NOTFOUND
 
//Path to a file.
WEBP_INCLUDE_DIR:PATH=WEBP_INCLUDE_DIR-NOTFOUND
 
//Include IEEE1394 support
WITH_1394:BOOL=ON
 
//Include Clp support (EPL)
WITH_CLP:BOOL=OFF
 
//Include C= support
WITH_CSTRIPES:BOOL=OFF
 
//Include NVidia Cuda Basic Linear Algebra Subprograms (BLAS) library
// support
WITH_CUBLAS:BOOL=OFF
 
//Include NVidia Cuda Runtime support
WITH_CUDA:BOOL=ON
 
//Include NVidia Cuda Fast Fourier Transform (FFT) library support
WITH_CUFFT:BOOL=ON
 
//Include DirectX support
WITH_DIRECTX:BOOL=ON
 
//Build VideoIO with DirectShow support
WITH_DSHOW:BOOL=ON
 
//Include Eigen2/Eigen3 support
WITH_EIGEN:BOOL=ON
 
//Include FFMPEG support
WITH_FFMPEG:BOOL=ON
 
//Include GDAL Support
WITH_GDAL:BOOL=OFF
 
//Include DICOM support
WITH_GDCM:BOOL=OFF
 
//Include Smartek GigE support
WITH_GIGEAPI:BOOL=OFF
 
//Include Gstreamer support
WITH_GSTREAMER:BOOL=ON
 
//Enable Gstreamer 0.10 support (instead of 1.x)
WITH_GSTREAMER_0_10:BOOL=OFF
 
//Include Intel Perceptual Computing support
WITH_INTELPERC:BOOL=OFF
 
//Include Intel IPP support
WITH_IPP:BOOL=OFF
 
//Include Intel IPP_A support
WITH_IPP_A:BOOL=OFF
 
//Include JPEG2K support
WITH_JASPER:BOOL=ON
 
//Include JPEG support
WITH_JPEG:BOOL=ON
 
//Include Lapack library support
WITH_LAPACK:BOOL=ON
 
//Include Matlab support
WITH_MATLAB:BOOL=ON
 
//Build VideoIO with Media Foundation support
WITH_MSMF:BOOL=OFF
 
//Include NVidia Video Decoding library support
WITH_NVCUVID:BOOL=OFF
 
//Include OpenCL Runtime support
WITH_OPENCL:BOOL=ON
 
//Include AMD OpenCL BLAS library support
WITH_OPENCLAMDBLAS:BOOL=ON
 
//Include AMD OpenCL FFT library support
WITH_OPENCLAMDFFT:BOOL=ON
 
//Include OpenCL Shared Virtual Memory support
WITH_OPENCL_SVM:BOOL=OFF
 
//Include ILM support via OpenEXR
WITH_OPENEXR:BOOL=ON
 
//Include OpenGL support
WITH_OPENGL:BOOL=OFF
 
//Include OpenMP support
WITH_OPENMP:BOOL=OFF
 
//Include OpenNI support
WITH_OPENNI:BOOL=OFF
 
//Include OpenNI2 support
WITH_OPENNI2:BOOL=OFF
 
//Include OpenVX support
WITH_OPENVX:BOOL=OFF
 
//Include PNG support
WITH_PNG:BOOL=ON
 
//Use pthreads-based parallel_for
WITH_PTHREADS_PF:BOOL=ON
 
//Include Prosilica GigE support
WITH_PVAPI:BOOL=OFF
 
//Build with Qt Backend support
WITH_QT:BOOL=OFF
 
//Include Intel TBB support
WITH_TBB:BOOL=OFF
 
//Include TIFF support
WITH_TIFF:BOOL=ON
 
//Include Video for Windows support
WITH_VFW:BOOL=ON
 
//Include VTK library support (and build opencv_viz module eiher)
WITH_VTK:BOOL=ON
 
//Include WebP support
WITH_WEBP:BOOL=ON
 
//Build with Win32 UI Backend support
WITH_WIN32UI:BOOL=ON
 
//Include XIMEA cameras support
WITH_XIMEA:BOOL=OFF
 
//Value Computed by CMake
annotation_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/apps/annotation
 
//Value Computed by CMake
annotation_SOURCE_DIR:STATIC=F:/opencv/sources/apps/annotation
 
//Value Computed by CMake
createsamples_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/apps/createsamples
 
//Value Computed by CMake
createsamples_SOURCE_DIR:STATIC=F:/opencv/sources/apps/createsamples
 
//Value Computed by CMake
libjasper_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/3rdparty/libjasper
 
//Dependencies for target
libjasper_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
libjasper_SOURCE_DIR:STATIC=F:/opencv/sources/3rdparty/libjasper
 
//Value Computed by CMake
libjpeg_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/3rdparty/libjpeg
 
//Dependencies for target
libjpeg_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
libjpeg_SOURCE_DIR:STATIC=F:/opencv/sources/3rdparty/libjpeg
 
//Value Computed by CMake
libpng_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/3rdparty/libpng
 
//Dependencies for the target
libpng_LIB_DEPENDS:STATIC=general;zlib;
 
//Value Computed by CMake
libpng_SOURCE_DIR:STATIC=F:/opencv/sources/3rdparty/libpng
 
//Value Computed by CMake
libtiff_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/3rdparty/libtiff
 
//Dependencies for the target
libtiff_LIB_DEPENDS:STATIC=general;zlib;
 
//Value Computed by CMake
libtiff_SOURCE_DIR:STATIC=F:/opencv/sources/3rdparty/libtiff
 
//Value Computed by CMake
libwebp_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/3rdparty/libwebp
 
//Dependencies for target
libwebp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
libwebp_SOURCE_DIR:STATIC=F:/opencv/sources/3rdparty/libwebp
 
//Value Computed by CMake
opencv_calib3d_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/calib3d
 
//Dependencies for the target
opencv_calib3d_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_flann;general;opencv_imgproc;general;opencv_ml;general;opencv_imgcodecs;general;opencv_videoio;general;opencv_highgui;general;opencv_features2d;
 
//Value Computed by CMake
opencv_calib3d_SOURCE_DIR:STATIC=F:/opencv/sources/modules/calib3d
 
//Dependencies for target
opencv_calib3d_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_core_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/core
 
//Dependencies for the target
opencv_core_LIB_DEPENDS:STATIC=general;zlib;
 
//Value Computed by CMake
opencv_core_SOURCE_DIR:STATIC=F:/opencv/sources/modules/core
 
//Dependencies for target
opencv_core_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_features2d_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/features2d
 
//Dependencies for the target
opencv_features2d_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_flann;general;opencv_imgproc;general;opencv_ml;general;opencv_imgcodecs;general;opencv_videoio;general;opencv_highgui;
 
//Value Computed by CMake
opencv_features2d_SOURCE_DIR:STATIC=F:/opencv/sources/modules/features2d
 
//Dependencies for target
opencv_features2d_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_flann_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/flann
 
//Dependencies for the target
opencv_flann_LIB_DEPENDS:STATIC=general;opencv_core;
 
//Value Computed by CMake
opencv_flann_SOURCE_DIR:STATIC=F:/opencv/sources/modules/flann
 
//Dependencies for target
opencv_flann_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_highgui_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/highgui
 
//Dependencies for the target
opencv_highgui_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_imgproc;general;opencv_imgcodecs;general;opencv_videoio;general;comctl32;general;gdi32;general;ole32;general;setupapi;general;ws2_32;
 
//Value Computed by CMake
opencv_highgui_SOURCE_DIR:STATIC=F:/opencv/sources/modules/highgui
 
//Dependencies for target
opencv_highgui_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_imgcodecs_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/imgcodecs
 
//Dependencies for the target
opencv_imgcodecs_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_imgproc;general;zlib;general;libjpeg;general;libwebp;general;libpng;general;libtiff;general;libjasper;general;IlmImf;
 
//Value Computed by CMake
opencv_imgcodecs_SOURCE_DIR:STATIC=F:/opencv/sources/modules/imgcodecs
 
//Dependencies for target
opencv_imgcodecs_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_imgproc_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/imgproc
 
//Dependencies for the target
opencv_imgproc_LIB_DEPENDS:STATIC=general;opencv_core;
 
//Value Computed by CMake
opencv_imgproc_SOURCE_DIR:STATIC=F:/opencv/sources/modules/imgproc
 
//Dependencies for target
opencv_imgproc_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_java_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/java
 
//Dependencies for the target
opencv_java_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_flann;general;opencv_imgproc;general;opencv_ml;general;opencv_photo;general;opencv_video;general;opencv_imgcodecs;general;opencv_videoio;general;opencv_highgui;general;opencv_objdetect;general;opencv_features2d;general;opencv_calib3d;
 
//Value Computed by CMake
opencv_java_SOURCE_DIR:STATIC=F:/opencv/sources/modules/java
 
//Value Computed by CMake
opencv_ml_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/ml
 
//Dependencies for the target
opencv_ml_LIB_DEPENDS:STATIC=general;opencv_core;
 
//Value Computed by CMake
opencv_ml_SOURCE_DIR:STATIC=F:/opencv/sources/modules/ml
 
//Dependencies for target
opencv_ml_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_objdetect_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/objdetect
 
//Dependencies for the target
opencv_objdetect_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_imgproc;general;opencv_ml;general;opencv_imgcodecs;general;opencv_videoio;general;opencv_highgui;
 
//Value Computed by CMake
opencv_objdetect_SOURCE_DIR:STATIC=F:/opencv/sources/modules/objdetect
 
//Dependencies for target
opencv_objdetect_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_calib3d_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_core_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_features2d_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_imgcodecs_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_imgproc_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_objdetect_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_photo_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_stitching_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_superres_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_video_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_perf_videoio_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_photo_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/photo
 
//Dependencies for the target
opencv_photo_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_imgproc;
 
//Value Computed by CMake
opencv_photo_SOURCE_DIR:STATIC=F:/opencv/sources/modules/photo
 
//Dependencies for target
opencv_photo_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_shape_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/shape
 
//Dependencies for the target
opencv_shape_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_imgproc;general;opencv_video;
 
//Value Computed by CMake
opencv_shape_SOURCE_DIR:STATIC=F:/opencv/sources/modules/shape
 
//Dependencies for target
opencv_shape_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_stitching_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/stitching
 
//Dependencies for the target
opencv_stitching_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_flann;general;opencv_imgproc;general;opencv_ml;general;opencv_imgcodecs;general;opencv_videoio;general;opencv_highgui;general;opencv_objdetect;general;opencv_features2d;general;opencv_calib3d;
 
//Value Computed by CMake
opencv_stitching_SOURCE_DIR:STATIC=F:/opencv/sources/modules/stitching
 
//Dependencies for target
opencv_stitching_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_superres_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/superres
 
//Dependencies for the target
opencv_superres_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_imgproc;general;opencv_video;general;opencv_imgcodecs;general;opencv_videoio;
 
//Value Computed by CMake
opencv_superres_SOURCE_DIR:STATIC=F:/opencv/sources/modules/superres
 
//Dependencies for target
opencv_superres_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_calib3d_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_core_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_features2d_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_flann_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_highgui_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_imgcodecs_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_imgproc_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_test_java_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/java/pure_test
 
//Value Computed by CMake
opencv_test_java_SOURCE_DIR:STATIC=F:/opencv/sources/modules/java/pure_test
 
//Dependencies for target
opencv_test_ml_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_objdetect_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_photo_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_shape_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_stitching_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_superres_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_video_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Dependencies for target
opencv_test_videoio_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_ts_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/ts
 
//Dependencies for the target
opencv_ts_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_imgproc;general;opencv_imgcodecs;general;opencv_videoio;general;opencv_highgui;
 
//Value Computed by CMake
opencv_ts_SOURCE_DIR:STATIC=F:/opencv/sources/modules/ts
 
//Dependencies for target
opencv_ts_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_version_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/apps/version
 
//Value Computed by CMake
opencv_version_SOURCE_DIR:STATIC=F:/opencv/sources/apps/version
 
//Value Computed by CMake
opencv_video_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/video
 
//Dependencies for the target
opencv_video_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_imgproc;
 
//Value Computed by CMake
opencv_video_SOURCE_DIR:STATIC=F:/opencv/sources/modules/video
 
//Dependencies for target
opencv_video_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_videoio_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/videoio
 
//Dependencies for the target
opencv_videoio_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_imgproc;general;opencv_imgcodecs;general;vfw32;general;winmm;
 
//Value Computed by CMake
opencv_videoio_SOURCE_DIR:STATIC=F:/opencv/sources/modules/videoio
 
//Dependencies for target
opencv_videoio_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
opencv_videostab_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/modules/videostab
 
//Dependencies for the target
opencv_videostab_LIB_DEPENDS:STATIC=general;opencv_core;general;opencv_flann;general;opencv_imgproc;general;opencv_ml;general;opencv_photo;general;opencv_video;general;opencv_imgcodecs;general;opencv_videoio;general;opencv_highgui;general;opencv_features2d;general;opencv_calib3d;
 
//Value Computed by CMake
opencv_videostab_SOURCE_DIR:STATIC=F:/opencv/sources/modules/videostab
 
//Dependencies for target
opencv_videostab_pch_dephelp_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
openexr_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/3rdparty/openexr
 
//Value Computed by CMake
openexr_SOURCE_DIR:STATIC=F:/opencv/sources/3rdparty/openexr
 
//Value Computed by CMake
traincascade_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/apps/traincascade
 
//Value Computed by CMake
traincascade_SOURCE_DIR:STATIC=F:/opencv/sources/apps/traincascade
 
//Value Computed by CMake
visualisation_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/apps/visualisation
 
//Value Computed by CMake
visualisation_SOURCE_DIR:STATIC=F:/opencv/sources/apps/visualisation
 
//Value Computed by CMake
zlib_BINARY_DIR:STATIC=F:/opencv/sources/opencv3/3rdparty/zlib
 
//Dependencies for target
zlib_LIB_DEPENDS:STATIC=
 
//Value Computed by CMake
zlib_SOURCE_DIR:STATIC=F:/opencv/sources/3rdparty/zlib
 
 
########################
# INTERNAL cache entries
########################
 
//Detected ant vesion
ANT_VERSION:INTERNAL=1.10.1
//ADVANCED property for variable: CMAKE_AR
CMAKE_AR-ADVANCED:INTERNAL=1
//This is the directory where this CMakeCache.txt was created
CMAKE_CACHEFILE_DIR:INTERNAL=f:/opencv/sources/opencv3
//Major version of cmake used to create the current loaded cache
CMAKE_CACHE_MAJOR_VERSION:INTERNAL=3
//Minor version of cmake used to create the current loaded cache
CMAKE_CACHE_MINOR_VERSION:INTERNAL=8
//Patch version of cmake used to create the current loaded cache
CMAKE_CACHE_PATCH_VERSION:INTERNAL=0
//ADVANCED property for variable: CMAKE_COLOR_MAKEFILE
CMAKE_COLOR_MAKEFILE-ADVANCED:INTERNAL=1
//Path to CMake executable.
CMAKE_COMMAND:INTERNAL=C:/Program Files/CMake/bin/cmake.exe
//Path to cpack program executable.
CMAKE_CPACK_COMMAND:INTERNAL=C:/Program Files/CMake/bin/cpack.exe
//Path to ctest program executable.
CMAKE_CTEST_COMMAND:INTERNAL=C:/Program Files/CMake/bin/ctest.exe
//ADVANCED property for variable: CMAKE_CXX_COMPILER
CMAKE_CXX_COMPILER-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_CXX_FLAGS
CMAKE_CXX_FLAGS-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_CXX_FLAGS_DEBUG
CMAKE_CXX_FLAGS_DEBUG-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_CXX_FLAGS_MINSIZEREL
CMAKE_CXX_FLAGS_MINSIZEREL-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_CXX_FLAGS_RELEASE
CMAKE_CXX_FLAGS_RELEASE-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_CXX_FLAGS_RELWITHDEBINFO
CMAKE_CXX_FLAGS_RELWITHDEBINFO-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_CXX_STANDARD_LIBRARIES
CMAKE_CXX_STANDARD_LIBRARIES-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_C_COMPILER
CMAKE_C_COMPILER-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_C_FLAGS
CMAKE_C_FLAGS-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_C_FLAGS_DEBUG
CMAKE_C_FLAGS_DEBUG-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_C_FLAGS_MINSIZEREL
CMAKE_C_FLAGS_MINSIZEREL-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_C_FLAGS_RELEASE
CMAKE_C_FLAGS_RELEASE-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_C_FLAGS_RELWITHDEBINFO
CMAKE_C_FLAGS_RELWITHDEBINFO-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_C_STANDARD_LIBRARIES
CMAKE_C_STANDARD_LIBRARIES-ADVANCED:INTERNAL=1
//Path to cache edit program executable.
CMAKE_EDIT_COMMAND:INTERNAL=C:/Program Files/CMake/bin/cmake-gui.exe
//Executable file format
CMAKE_EXECUTABLE_FORMAT:INTERNAL=Unknown
//ADVANCED property for variable: CMAKE_EXE_LINKER_FLAGS
CMAKE_EXE_LINKER_FLAGS-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_EXE_LINKER_FLAGS_DEBUG
CMAKE_EXE_LINKER_FLAGS_DEBUG-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_EXE_LINKER_FLAGS_MINSIZEREL
CMAKE_EXE_LINKER_FLAGS_MINSIZEREL-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_EXE_LINKER_FLAGS_RELEASE
CMAKE_EXE_LINKER_FLAGS_RELEASE-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO
CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO-ADVANCED:INTERNAL=1
//Name of external makefile project generator.
CMAKE_EXTRA_GENERATOR:INTERNAL=
//Name of generator.
CMAKE_GENERATOR:INTERNAL=MinGW Makefiles
//Name of generator platform.
CMAKE_GENERATOR_PLATFORM:INTERNAL=
//Name of generator toolset.
CMAKE_GENERATOR_TOOLSET:INTERNAL=
//Source directory with the top level CMakeLists.txt file for this
// project
CMAKE_HOME_DIRECTORY:INTERNAL=F:/opencv/sources
//ADVANCED property for variable: CMAKE_LINKER
CMAKE_LINKER-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_MAKE_PROGRAM
CMAKE_MAKE_PROGRAM-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_MODULE_LINKER_FLAGS
CMAKE_MODULE_LINKER_FLAGS-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_MODULE_LINKER_FLAGS_DEBUG
CMAKE_MODULE_LINKER_FLAGS_DEBUG-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL
CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_MODULE_LINKER_FLAGS_RELEASE
CMAKE_MODULE_LINKER_FLAGS_RELEASE-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO
CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_NM
CMAKE_NM-ADVANCED:INTERNAL=1
//number of local generators
CMAKE_NUMBER_OF_MAKEFILES:INTERNAL=73
//ADVANCED property for variable: CMAKE_OBJCOPY
CMAKE_OBJCOPY-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_OBJDUMP
CMAKE_OBJDUMP-ADVANCED:INTERNAL=1
//Platform information initialized
CMAKE_PLATFORM_INFO_INITIALIZED:INTERNAL=1
//ADVANCED property for variable: CMAKE_RANLIB
CMAKE_RANLIB-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_RC_COMPILER
CMAKE_RC_COMPILER-ADVANCED:INTERNAL=1
CMAKE_RC_COMPILER_WORKS:INTERNAL=1
//ADVANCED property for variable: CMAKE_RC_FLAGS
CMAKE_RC_FLAGS-ADVANCED:INTERNAL=1
//Path to CMake installation.
CMAKE_ROOT:INTERNAL=C:/Program Files/CMake/share/cmake-3.8
//ADVANCED property for variable: CMAKE_SH
CMAKE_SH-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_SHARED_LINKER_FLAGS
CMAKE_SHARED_LINKER_FLAGS-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_SHARED_LINKER_FLAGS_DEBUG
CMAKE_SHARED_LINKER_FLAGS_DEBUG-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL
CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_SHARED_LINKER_FLAGS_RELEASE
CMAKE_SHARED_LINKER_FLAGS_RELEASE-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO
CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO-ADVANCED:INTERNAL=1
//CHECK_TYPE_SIZE: sizeof(unsigned short)
CMAKE_SIZEOF_UNSIGNED_SHORT:INTERNAL=2
//ADVANCED property for variable: CMAKE_SKIP_INSTALL_RPATH
CMAKE_SKIP_INSTALL_RPATH-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_SKIP_RPATH
CMAKE_SKIP_RPATH-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_STATIC_LINKER_FLAGS
CMAKE_STATIC_LINKER_FLAGS-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_STATIC_LINKER_FLAGS_DEBUG
CMAKE_STATIC_LINKER_FLAGS_DEBUG-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_STATIC_LINKER_FLAGS_MINSIZEREL
CMAKE_STATIC_LINKER_FLAGS_MINSIZEREL-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_STATIC_LINKER_FLAGS_RELEASE
CMAKE_STATIC_LINKER_FLAGS_RELEASE-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO
CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_STRIP
CMAKE_STRIP-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CMAKE_VERBOSE_MAKEFILE
CMAKE_VERBOSE_MAKEFILE-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CPACK_BINARY_7Z
CPACK_BINARY_7Z-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CPACK_BINARY_IFW
CPACK_BINARY_IFW-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CPACK_BINARY_NSIS
CPACK_BINARY_NSIS-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CPACK_BINARY_WIX
CPACK_BINARY_WIX-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CPACK_BINARY_ZIP
CPACK_BINARY_ZIP-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CPACK_SOURCE_7Z
CPACK_SOURCE_7Z-ADVANCED:INTERNAL=1
//ADVANCED property for variable: CPACK_SOURCE_ZIP
CPACK_SOURCE_ZIP-ADVANCED:INTERNAL=1
//ADVANCED property for variable: DOXYGEN_DOT_EXECUTABLE
DOXYGEN_DOT_EXECUTABLE-ADVANCED:INTERNAL=1
//ADVANCED property for variable: DOXYGEN_EXECUTABLE
DOXYGEN_EXECUTABLE-ADVANCED:INTERNAL=1
//Details about finding JNI
FIND_PACKAGE_MESSAGE_DETAILS_JNI:INTERNAL=[C:/Program Files/Java/jdk1.8.0_45/lib/jawt.lib][C:/Program Files/Java/jdk1.8.0_45/lib/jvm.lib][C:/Program Files/Java/jdk1.8.0_45/include][C:/Program Files/Java/jdk1.8.0_45/include/win32][C:/Program Files/Java/jdk1.8.0_45/include][v()]
//Details about finding Matlab
FIND_PACKAGE_MESSAGE_DETAILS_Matlab:INTERNAL=[D:/Program Files/MATLAB/R2016b/bin/mex.bat][D:/Program Files/MATLAB/R2016b/extern/include][D:/Program Files/MATLAB/R2016b][D:/Program Files/MATLAB/R2016b/bin/win64/libmx.dll;D:/Program Files/MATLAB/R2016b/bin/win64/libmex.dll;D:/Program Files/MATLAB/R2016b/bin/win64/libmat.dll][D:/Program Files/MATLAB/R2016b/bin/win64][mexw64][win64][D:/Program Files/MATLAB/R2016b/bin/matlab.exe][v()]
//Details about finding PythonInterp
FIND_PACKAGE_MESSAGE_DETAILS_PythonInterp:INTERNAL=[C:/yotta/python/python.exe][v2.7.10(2.7)]
//Details about finding PythonLibs
FIND_PACKAGE_MESSAGE_DETAILS_PythonLibs:INTERNAL=[C:/yotta/python/libs/libpython27.a][C:/yotta/python/include][v2.7.10+(2.7.10)]
//ADVANCED property for variable: GIT_EXECUTABLE
GIT_EXECUTABLE-ADVANCED:INTERNAL=1
//Have include assert.h
HAVE_ASSERT_H:INTERNAL=1
//Result of TRY_COMPILE
HAVE_CMAKE_SIZEOF_UNSIGNED_SHORT:INTERNAL=TRUE
//Test HAVE_CXX_FDIAGNOSTICS_SHOW_OPTION
HAVE_CXX_FDIAGNOSTICS_SHOW_OPTION:INTERNAL=1
//Test HAVE_CXX_FFUNCTION_SECTIONS
HAVE_CXX_FFUNCTION_SECTIONS:INTERNAL=1
//Test HAVE_CXX_FOMIT_FRAME_POINTER
HAVE_CXX_FOMIT_FRAME_POINTER:INTERNAL=1
//Test HAVE_CXX_FSIGNED_CHAR
HAVE_CXX_FSIGNED_CHAR:INTERNAL=1
//Test HAVE_CXX_FVISIBILITY_HIDDEN
HAVE_CXX_FVISIBILITY_HIDDEN:INTERNAL=1
//Test HAVE_CXX_FVISIBILITY_INLINES_HIDDEN
HAVE_CXX_FVISIBILITY_INLINES_HIDDEN:INTERNAL=1
//Test HAVE_CXX_MARCH_I686
HAVE_CXX_MARCH_I686:INTERNAL=1
//Test HAVE_CXX_MFPMATH_SSE
HAVE_CXX_MFPMATH_SSE:INTERNAL=1
//Test HAVE_CXX_MSSE
HAVE_CXX_MSSE:INTERNAL=1
//Test HAVE_CXX_MSSE2
HAVE_CXX_MSSE2:INTERNAL=1
//Test HAVE_CXX_MSTACKREALIGN
HAVE_CXX_MSTACKREALIGN:INTERNAL=1
//Test HAVE_CXX_W
HAVE_CXX_W:INTERNAL=1
//Test HAVE_CXX_WALL
HAVE_CXX_WALL:INTERNAL=1
//Test HAVE_CXX_WERROR_ADDRESS
HAVE_CXX_WERROR_ADDRESS:INTERNAL=1
//Test HAVE_CXX_WERROR_FORMAT_SECURITY
HAVE_CXX_WERROR_FORMAT_SECURITY:INTERNAL=1
//Test HAVE_CXX_WERROR_NON_VIRTUAL_DTOR
HAVE_CXX_WERROR_NON_VIRTUAL_DTOR:INTERNAL=1
//Test HAVE_CXX_WERROR_RETURN_TYPE
HAVE_CXX_WERROR_RETURN_TYPE:INTERNAL=1
//Test HAVE_CXX_WERROR_SEQUENCE_POINT
HAVE_CXX_WERROR_SEQUENCE_POINT:INTERNAL=1
//Test HAVE_CXX_WFORMAT
HAVE_CXX_WFORMAT:INTERNAL=1
//Test HAVE_CXX_WINIT_SELF
HAVE_CXX_WINIT_SELF:INTERNAL=1
//Test HAVE_CXX_WMISSING_DECLARATIONS
HAVE_CXX_WMISSING_DECLARATIONS:INTERNAL=1
//Test HAVE_CXX_WMISSING_PROTOTYPES
HAVE_CXX_WMISSING_PROTOTYPES:INTERNAL=
//Test HAVE_CXX_WNO_ARRAY_BOUNDS
HAVE_CXX_WNO_ARRAY_BOUNDS:INTERNAL=1
//Test HAVE_CXX_WNO_COMMENT
HAVE_CXX_WNO_COMMENT:INTERNAL=1
//Test HAVE_CXX_WNO_DELETE_NON_VIRTUAL_DTOR
HAVE_CXX_WNO_DELETE_NON_VIRTUAL_DTOR:INTERNAL=1
//Test HAVE_CXX_WNO_DEPRECATED_DECLARATIONS
HAVE_CXX_WNO_DEPRECATED_DECLARATIONS:INTERNAL=1
//Test HAVE_CXX_WNO_EXTRA
HAVE_CXX_WNO_EXTRA:INTERNAL=1
//Test HAVE_CXX_WNO_MISLEADING_INDENTATION
HAVE_CXX_WNO_MISLEADING_INDENTATION:INTERNAL=
//Test HAVE_CXX_WNO_MISSING_DECLARATIONS
HAVE_CXX_WNO_MISSING_DECLARATIONS:INTERNAL=1
//Test HAVE_CXX_WNO_NARROWING
HAVE_CXX_WNO_NARROWING:INTERNAL=1
//Test HAVE_CXX_WNO_OVERLOADED_VIRTUAL
HAVE_CXX_WNO_OVERLOADED_VIRTUAL:INTERNAL=1
//Test HAVE_CXX_WNO_PARENTHESES
HAVE_CXX_WNO_PARENTHESES:INTERNAL=1
//Test HAVE_CXX_WNO_SHADOW
HAVE_CXX_WNO_SHADOW:INTERNAL=1
//Test HAVE_CXX_WNO_SIGN_COMPARE
HAVE_CXX_WNO_SIGN_COMPARE:INTERNAL=1
//Test HAVE_CXX_WNO_SWITCH
HAVE_CXX_WNO_SWITCH:INTERNAL=1
//Test HAVE_CXX_WNO_UNDEF
HAVE_CXX_WNO_UNDEF:INTERNAL=1
//Test HAVE_CXX_WNO_UNINITIALIZED
HAVE_CXX_WNO_UNINITIALIZED:INTERNAL=1
//Test HAVE_CXX_WNO_UNNAMED_TYPE_TEMPLATE_ARGS
HAVE_CXX_WNO_UNNAMED_TYPE_TEMPLATE_ARGS:INTERNAL=
//Test HAVE_CXX_WNO_UNUSED
HAVE_CXX_WNO_UNUSED:INTERNAL=1
//Test HAVE_CXX_WNO_UNUSED_CONST_VARIABLE
HAVE_CXX_WNO_UNUSED_CONST_VARIABLE:INTERNAL=
//Test HAVE_CXX_WNO_UNUSED_PARAMETER
HAVE_CXX_WNO_UNUSED_PARAMETER:INTERNAL=1
//Test HAVE_CXX_WPOINTER_ARITH
HAVE_CXX_WPOINTER_ARITH:INTERNAL=1
//Test HAVE_CXX_WSHADOW
HAVE_CXX_WSHADOW:INTERNAL=1
//Test HAVE_CXX_WSIGN_PROMO
HAVE_CXX_WSIGN_PROMO:INTERNAL=1
//Test HAVE_CXX_WSTRICT_PROTOTYPES
HAVE_CXX_WSTRICT_PROTOTYPES:INTERNAL=
//Test HAVE_CXX_WUNDEF
HAVE_CXX_WUNDEF:INTERNAL=1
//Test HAVE_C_FDIAGNOSTICS_SHOW_OPTION
HAVE_C_FDIAGNOSTICS_SHOW_OPTION:INTERNAL=1
//Test HAVE_C_FFUNCTION_SECTIONS
HAVE_C_FFUNCTION_SECTIONS:INTERNAL=1
//Test HAVE_C_FOMIT_FRAME_POINTER
HAVE_C_FOMIT_FRAME_POINTER:INTERNAL=1
//Test HAVE_C_FSIGNED_CHAR
HAVE_C_FSIGNED_CHAR:INTERNAL=1
//Test HAVE_C_FVISIBILITY_HIDDEN
HAVE_C_FVISIBILITY_HIDDEN:INTERNAL=1
//Test HAVE_C_FVISIBILITY_INLINES_HIDDEN
HAVE_C_FVISIBILITY_INLINES_HIDDEN:INTERNAL=
//Test HAVE_C_MARCH_I686
HAVE_C_MARCH_I686:INTERNAL=1
//Test HAVE_C_MFPMATH_SSE
HAVE_C_MFPMATH_SSE:INTERNAL=1
//Test HAVE_C_MSSE
HAVE_C_MSSE:INTERNAL=1
//Test HAVE_C_MSSE2
HAVE_C_MSSE2:INTERNAL=1
//Test HAVE_C_MSTACKREALIGN
HAVE_C_MSTACKREALIGN:INTERNAL=1
//Test HAVE_C_W
HAVE_C_W:INTERNAL=1
//Test HAVE_C_WALL
HAVE_C_WALL:INTERNAL=1
//Test HAVE_C_WERROR_ADDRESS
HAVE_C_WERROR_ADDRESS:INTERNAL=1
//Test HAVE_C_WERROR_FORMAT_SECURITY
HAVE_C_WERROR_FORMAT_SECURITY:INTERNAL=1
//Test HAVE_C_WERROR_NON_VIRTUAL_DTOR
HAVE_C_WERROR_NON_VIRTUAL_DTOR:INTERNAL=1
//Test HAVE_C_WERROR_RETURN_TYPE
HAVE_C_WERROR_RETURN_TYPE:INTERNAL=1
//Test HAVE_C_WERROR_SEQUENCE_POINT
HAVE_C_WERROR_SEQUENCE_POINT:INTERNAL=1
//Test HAVE_C_WFORMAT
HAVE_C_WFORMAT:INTERNAL=1
//Test HAVE_C_WINIT_SELF
HAVE_C_WINIT_SELF:INTERNAL=1
//Test HAVE_C_WMISSING_DECLARATIONS
HAVE_C_WMISSING_DECLARATIONS:INTERNAL=1
//Test HAVE_C_WMISSING_PROTOTYPES
HAVE_C_WMISSING_PROTOTYPES:INTERNAL=1
//Test HAVE_C_WNO_ATTRIBUTES
HAVE_C_WNO_ATTRIBUTES:INTERNAL=1
//Test HAVE_C_WNO_CAST_ALIGN
HAVE_C_WNO_CAST_ALIGN:INTERNAL=1
//Test HAVE_C_WNO_COMMENT
HAVE_C_WNO_COMMENT:INTERNAL=1
//Test HAVE_C_WNO_DELETE_NON_VIRTUAL_DTOR
HAVE_C_WNO_DELETE_NON_VIRTUAL_DTOR:INTERNAL=
//Test HAVE_C_WNO_IMPLICIT_FUNCTION_DECLARATION
HAVE_C_WNO_IMPLICIT_FUNCTION_DECLARATION:INTERNAL=1
//Test HAVE_C_WNO_INT_TO_POINTER_CAST
HAVE_C_WNO_INT_TO_POINTER_CAST:INTERNAL=1
//Test HAVE_C_WNO_MAYBE_UNINITIALIZED
HAVE_C_WNO_MAYBE_UNINITIALIZED:INTERNAL=1
//Test HAVE_C_WNO_MISLEADING_INDENTATION
HAVE_C_WNO_MISLEADING_INDENTATION:INTERNAL=
//Test HAVE_C_WNO_MISSING_DECLARATIONS
HAVE_C_WNO_MISSING_DECLARATIONS:INTERNAL=1
//Test HAVE_C_WNO_MISSING_PROTOTYPES
HAVE_C_WNO_MISSING_PROTOTYPES:INTERNAL=1
//Test HAVE_C_WNO_NARROWING
HAVE_C_WNO_NARROWING:INTERNAL=1
//Test HAVE_C_WNO_POINTER_TO_INT_CAST
HAVE_C_WNO_POINTER_TO_INT_CAST:INTERNAL=1
//Test HAVE_C_WNO_SHADOW
HAVE_C_WNO_SHADOW:INTERNAL=1
//Test HAVE_C_WNO_SHIFT_NEGATIVE_VALUE
HAVE_C_WNO_SHIFT_NEGATIVE_VALUE:INTERNAL=
//Test HAVE_C_WNO_SHORTEN_64_TO_32
HAVE_C_WNO_SHORTEN_64_TO_32:INTERNAL=
//Test HAVE_C_WNO_SIGN_COMPARE
HAVE_C_WNO_SIGN_COMPARE:INTERNAL=1
//Test HAVE_C_WNO_STRICT_OVERFLOW
HAVE_C_WNO_STRICT_OVERFLOW:INTERNAL=1
//Test HAVE_C_WNO_STRICT_PROTOTYPES
HAVE_C_WNO_STRICT_PROTOTYPES:INTERNAL=1
//Test HAVE_C_WNO_UNDEF
HAVE_C_WNO_UNDEF:INTERNAL=1
//Test HAVE_C_WNO_UNINITIALIZED
HAVE_C_WNO_UNINITIALIZED:INTERNAL=1
//Test HAVE_C_WNO_UNNAMED_TYPE_TEMPLATE_ARGS
HAVE_C_WNO_UNNAMED_TYPE_TEMPLATE_ARGS:INTERNAL=
//Test HAVE_C_WNO_UNUSED
HAVE_C_WNO_UNUSED:INTERNAL=1
//Test HAVE_C_WNO_UNUSED_BUT_SET_PARAMETER
HAVE_C_WNO_UNUSED_BUT_SET_PARAMETER:INTERNAL=1
//Test HAVE_C_WNO_UNUSED_BUT_SET_VARIABLE
HAVE_C_WNO_UNUSED_BUT_SET_VARIABLE:INTERNAL=1
//Test HAVE_C_WNO_UNUSED_FUNCTION
HAVE_C_WNO_UNUSED_FUNCTION:INTERNAL=1
//Test HAVE_C_WNO_UNUSED_PARAMETER
HAVE_C_WNO_UNUSED_PARAMETER:INTERNAL=1
//Test HAVE_C_WNO_UNUSED_VARIABLE
HAVE_C_WNO_UNUSED_VARIABLE:INTERNAL=1
//Test HAVE_C_WPOINTER_ARITH
HAVE_C_WPOINTER_ARITH:INTERNAL=1
//Test HAVE_C_WSHADOW
HAVE_C_WSHADOW:INTERNAL=1
//Test HAVE_C_WSIGN_PROMO
HAVE_C_WSIGN_PROMO:INTERNAL=
//Test HAVE_C_WSTRICT_PROTOTYPES
HAVE_C_WSTRICT_PROTOTYPES:INTERNAL=1
//Test HAVE_C_WUNDEF
HAVE_C_WUNDEF:INTERNAL=1
//Have include fcntl.h
HAVE_FCNTL_H:INTERNAL=1
//Have function fseeko
HAVE_FSEEKO:INTERNAL=1
//Have include io.h
HAVE_IO_H:INTERNAL=1
//Have function jbg_newlen
HAVE_JBG_NEWLEN:INTERNAL=
//Have function mmap
HAVE_MMAP:INTERNAL=
//Result of TRY_COMPILE
HAVE_OFF64_T:INTERNAL=TRUE
//Result of TRY_COMPILE
HAVE_PTHREADS:INTERNAL=TRUE
//Have include search.h
HAVE_SEARCH_H:INTERNAL=1
//Have include stddef.h
HAVE_STDDEF_H:INTERNAL=1
//Have include stdint.h
HAVE_STDINT_H:INTERNAL=1
//Have include string.h
HAVE_STRING_H:INTERNAL=1
//Have include sys/types.h
HAVE_SYS_TYPES_H:INTERNAL=1
//Have include unistd.h
HAVE_UNISTD_H:INTERNAL=1
//Result of TRY_COMPILE
HAVE_VFW:INTERNAL=TRUE
//Result of TRY_COMPILE
HAVE_WIN32UI:INTERNAL=TRUE
//Result of TRY_COMPILE
HAVE_WORDS_BIGENDIAN:INTERNAL=TRUE
//Module opencv_calib3d will be built in current configuration
HAVE_opencv_calib3d:INTERNAL=ON
//Module opencv_core will be built in current configuration
HAVE_opencv_core:INTERNAL=ON
//Module opencv_cudaarithm can not be built in current configuration
HAVE_opencv_cudaarithm:INTERNAL=OFF
//Module opencv_cudabgsegm can not be built in current configuration
HAVE_opencv_cudabgsegm:INTERNAL=OFF
//Module opencv_cudacodec can not be built in current configuration
HAVE_opencv_cudacodec:INTERNAL=OFF
//Module opencv_cudafeatures2d can not be built in current configuration
HAVE_opencv_cudafeatures2d:INTERNAL=OFF
//Module opencv_cudafilters can not be built in current configuration
HAVE_opencv_cudafilters:INTERNAL=OFF
//Module opencv_cudaimgproc can not be built in current configuration
HAVE_opencv_cudaimgproc:INTERNAL=OFF
//Module opencv_cudalegacy can not be built in current configuration
HAVE_opencv_cudalegacy:INTERNAL=OFF
//Module opencv_cudaobjdetect can not be built in current configuration
HAVE_opencv_cudaobjdetect:INTERNAL=OFF
//Module opencv_cudaoptflow can not be built in current configuration
HAVE_opencv_cudaoptflow:INTERNAL=OFF
//Module opencv_cudastereo can not be built in current configuration
HAVE_opencv_cudastereo:INTERNAL=OFF
//Module opencv_cudawarping can not be built in current configuration
HAVE_opencv_cudawarping:INTERNAL=OFF
//Module opencv_cudev can not be built in current configuration
HAVE_opencv_cudev:INTERNAL=OFF
//Module opencv_features2d will be built in current configuration
HAVE_opencv_features2d:INTERNAL=ON
//Module opencv_flann will be built in current configuration
HAVE_opencv_flann:INTERNAL=ON
//Module opencv_highgui will be built in current configuration
HAVE_opencv_highgui:INTERNAL=ON
//Module opencv_imgcodecs will be built in current configuration
HAVE_opencv_imgcodecs:INTERNAL=ON
//Module opencv_imgproc will be built in current configuration
HAVE_opencv_imgproc:INTERNAL=ON
//Module opencv_java will be built in current configuration
HAVE_opencv_java:INTERNAL=ON
//Module opencv_ml will be built in current configuration
HAVE_opencv_ml:INTERNAL=ON
//Module opencv_objdetect will be built in current configuration
HAVE_opencv_objdetect:INTERNAL=ON
//Module opencv_photo will be built in current configuration
HAVE_opencv_photo:INTERNAL=ON
//Module opencv_python2 can not be built in current configuration
HAVE_opencv_python2:INTERNAL=OFF
//Module opencv_python3 can not be built in current configuration
HAVE_opencv_python3:INTERNAL=OFF
//Module opencv_shape will be built in current configuration
HAVE_opencv_shape:INTERNAL=ON
//Module opencv_stitching will be built in current configuration
HAVE_opencv_stitching:INTERNAL=ON
//Module opencv_superres will be built in current configuration
HAVE_opencv_superres:INTERNAL=ON
//Module opencv_ts will be built in current configuration
HAVE_opencv_ts:INTERNAL=ON
//Module opencv_video will be built in current configuration
HAVE_opencv_video:INTERNAL=ON
//Module opencv_videoio will be built in current configuration
HAVE_opencv_videoio:INTERNAL=ON
//Module opencv_videostab will be built in current configuration
HAVE_opencv_videostab:INTERNAL=ON
//Module opencv_viz can not be built in current configuration
HAVE_opencv_viz:INTERNAL=OFF
//Module opencv_world will not be built in current configuration
HAVE_opencv_world:INTERNAL=OFF
//ADVANCED property for variable: JAVA_AWT_INCLUDE_PATH
JAVA_AWT_INCLUDE_PATH-ADVANCED:INTERNAL=1
//ADVANCED property for variable: JAVA_AWT_LIBRARY
JAVA_AWT_LIBRARY-ADVANCED:INTERNAL=1
//ADVANCED property for variable: JAVA_INCLUDE_PATH
JAVA_INCLUDE_PATH-ADVANCED:INTERNAL=1
//ADVANCED property for variable: JAVA_INCLUDE_PATH2
JAVA_INCLUDE_PATH2-ADVANCED:INTERNAL=1
//ADVANCED property for variable: JAVA_JVM_LIBRARY
JAVA_JVM_LIBRARY-ADVANCED:INTERNAL=1
//CHECK_TYPE_SIZE: sizeof(off64_t)
OFF64_T:INTERNAL=8
//ADVANCED property for variable: OPENCL_INCLUDE_DIR
OPENCL_INCLUDE_DIR-ADVANCED:INTERNAL=1
//ADVANCED property for variable: OPENCL_LIBRARY
OPENCL_LIBRARY-ADVANCED:INTERNAL=1
//Extra compiler options for C++ sources
OPENCV_EXTRA_CXX_FLAGS:INTERNAL= -fsigned-char -mstackrealign -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wno-narrowing -Wno-delete-non-virtual-dtor -Wno-comment -fdiagnostics-show-option -march=i686 -fomit-frame-pointer -msse -msse2 -mfpmath=sse -ffunction-sections
//Extra compiler options for C sources
OPENCV_EXTRA_C_FLAGS:INTERNAL= -fsigned-char -mstackrealign -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wno-narrowing -Wno-comment -fdiagnostics-show-option -march=i686 -fomit-frame-pointer -msse -msse2 -mfpmath=sse -ffunction-sections
//Extra linker flags
OPENCV_EXTRA_EXE_LINKER_FLAGS:INTERNAL=
//Extra linker flags for Debug build
OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG:INTERNAL=
//Extra linker flags for Release build
OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE:INTERNAL=
//Extra compiler options
OPENCV_EXTRA_FLAGS:INTERNAL=
//Extra compiler options for Debug build
OPENCV_EXTRA_FLAGS_DEBUG:INTERNAL= -O0 -DDEBUG -D_DEBUG
//Extra compiler options for Release build
OPENCV_EXTRA_FLAGS_RELEASE:INTERNAL= -DNDEBUG
//List of OpenCV modules included into the build
OPENCV_MODULES_BUILD:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_photo;opencv_video;opencv_imgcodecs;opencv_shape;opencv_videoio;opencv_highgui;opencv_objdetect;opencv_superres;opencv_ts;opencv_features2d;opencv_calib3d;opencv_java;opencv_stitching;opencv_videostab
//List of OpenCV modules implicitly disabled due to dependencies
OPENCV_MODULES_DISABLED_AUTO:INTERNAL=
//List of OpenCV modules which can not be build in current configuration
OPENCV_MODULES_DISABLED_FORCE:INTERNAL=opencv_cudaarithm;opencv_cudabgsegm;opencv_cudacodec;opencv_cudafeatures2d;opencv_cudafilters;opencv_cudaimgproc;opencv_cudalegacy;opencv_cudaobjdetect;opencv_cudaoptflow;opencv_cudastereo;opencv_cudawarping;opencv_cudev;opencv_python2;opencv_python3;opencv_viz
//List of OpenCV modules explicitly disabled by user
OPENCV_MODULES_DISABLED_USER:INTERNAL=opencv_world
//List of OpenCV modules marked for export
OPENCV_MODULES_PUBLIC:INTERNAL=opencv_calib3d;opencv_core;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab
OPENCV_MODULE_opencv_calib3d_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/calib3d
//List of opencv_calib3d submodules
OPENCV_MODULE_opencv_calib3d_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_calib3d_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_calib3d module
OPENCV_MODULE_opencv_calib3d_DEPS:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_imgcodecs;opencv_videoio;opencv_highgui;opencv_features2d
//Extra dependencies of opencv_calib3d module
OPENCV_MODULE_opencv_calib3d_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_calib3d module (for linker)
OPENCV_MODULE_opencv_calib3d_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_imgcodecs;opencv_videoio;opencv_highgui;opencv_features2d
//Brief description of opencv_calib3d module
OPENCV_MODULE_opencv_calib3d_DESCRIPTION:INTERNAL=Camera Calibration and 3D Reconstruction
//List of header files for opencv_calib3d
OPENCV_MODULE_opencv_calib3d_HEADERS:INTERNAL=F:/opencv/sources/modules/calib3d/include/opencv2/calib3d.hpp;F:/opencv/sources/modules/calib3d/include/opencv2/calib3d/calib3d.hpp;F:/opencv/sources/modules/calib3d/include/opencv2/calib3d/calib3d_c.h
OPENCV_MODULE_opencv_calib3d_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_calib3d_LABEL:INTERNAL=Main;opencv_calib3d
OPENCV_MODULE_opencv_calib3d_LINK_DEPS:INTERNAL=;
//Location of opencv_calib3d module sources
OPENCV_MODULE_opencv_calib3d_LOCATION:INTERNAL=F:/opencv/sources/modules/calib3d
//Optional dependencies of opencv_calib3d module
OPENCV_MODULE_opencv_calib3d_OPT_DEPS:INTERNAL=
//Optional private dependencies of opencv_calib3d module
OPENCV_MODULE_opencv_calib3d_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_calib3d module
OPENCV_MODULE_opencv_calib3d_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_calib3d module
OPENCV_MODULE_opencv_calib3d_REQ_DEPS:INTERNAL=opencv_imgproc;opencv_features2d
//List of source files for opencv_calib3d
OPENCV_MODULE_opencv_calib3d_SOURCES:INTERNAL=F:/opencv/sources/modules/calib3d/src/calibinit.cpp;F:/opencv/sources/modules/calib3d/src/calibration.cpp;F:/opencv/sources/modules/calib3d/src/checkchessboard.cpp;F:/opencv/sources/modules/calib3d/src/circlesgrid.cpp;F:/opencv/sources/modules/calib3d/src/compat_ptsetreg.cpp;F:/opencv/sources/modules/calib3d/src/compat_stereo.cpp;F:/opencv/sources/modules/calib3d/src/dls.cpp;F:/opencv/sources/modules/calib3d/src/epnp.cpp;F:/opencv/sources/modules/calib3d/src/fisheye.cpp;F:/opencv/sources/modules/calib3d/src/five-point.cpp;F:/opencv/sources/modules/calib3d/src/fundam.cpp;F:/opencv/sources/modules/calib3d/src/homography_decomp.cpp;F:/opencv/sources/modules/calib3d/src/levmarq.cpp;F:/opencv/sources/modules/calib3d/src/main.cpp;F:/opencv/sources/modules/calib3d/src/p3p.cpp;F:/opencv/sources/modules/calib3d/src/polynom_solver.cpp;F:/opencv/sources/modules/calib3d/src/posit.cpp;F:/opencv/sources/modules/calib3d/src/ptsetreg.cpp;F:/opencv/sources/modules/calib3d/src/quadsubpix.cpp;F:/opencv/sources/modules/calib3d/src/rho.cpp;F:/opencv/sources/modules/calib3d/src/solvepnp.cpp;F:/opencv/sources/modules/calib3d/src/stereobm.cpp;F:/opencv/sources/modules/calib3d/src/stereosgbm.cpp;F:/opencv/sources/modules/calib3d/src/triangulate.cpp;F:/opencv/sources/modules/calib3d/src/upnp.cpp;F:/opencv/sources/modules/calib3d/src/opencl/stereobm.cl;F:/opencv/sources/opencv3/modules/calib3d/opencl_kernels_calib3d.cpp;F:/opencv/sources/opencv3/modules/calib3d/opencl_kernels_calib3d.hpp;F:/opencv/sources/modules/calib3d/src/circlesgrid.hpp;F:/opencv/sources/modules/calib3d/src/fisheye.hpp;F:/opencv/sources/modules/calib3d/src/precomp.hpp;F:/opencv/sources/modules/calib3d/src/dls.h;F:/opencv/sources/modules/calib3d/src/epnp.h;F:/opencv/sources/modules/calib3d/src/p3p.h;F:/opencv/sources/modules/calib3d/src/polynom_solver.h;F:/opencv/sources/modules/calib3d/src/rho.h;F:/opencv/sources/modules/calib3d/src/upnp.h
//List of wrappers supporting module opencv_calib3d
OPENCV_MODULE_opencv_calib3d_WRAPPERS:INTERNAL=java;python2;python3
OPENCV_MODULE_opencv_core_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/core
//List of opencv_core submodules
OPENCV_MODULE_opencv_core_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_core_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_core module
OPENCV_MODULE_opencv_core_DEPS:INTERNAL=
//Extra dependencies of opencv_core module
OPENCV_MODULE_opencv_core_DEPS_EXT:INTERNAL=zlib
//Flattened dependencies of opencv_core module (for linker)
OPENCV_MODULE_opencv_core_DEPS_TO_LINK:INTERNAL=
//Brief description of opencv_core module
OPENCV_MODULE_opencv_core_DESCRIPTION:INTERNAL=The Core Functionality
//List of header files for opencv_core
OPENCV_MODULE_opencv_core_HEADERS:INTERNAL=F:/opencv/sources/modules/core/include/opencv2/core/cuda/block.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/border_interpolate.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/color.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/common.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/datamov_utils.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/dynamic_smem.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/emulation.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/filters.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/funcattrib.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/functional.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/limits.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/reduce.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/saturate_cast.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/scan.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/simd_functions.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/transform.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/type_traits.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/utility.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/vec_distance.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/vec_math.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/vec_traits.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/warp.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/warp_reduce.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/warp_shuffle.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/detail/color_detail.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/detail/reduce.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/detail/reduce_key_val.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/detail/transform_detail.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/detail/type_traits_detail.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda/detail/vec_distance_detail.hpp;F:/opencv/sources/modules/core/include/opencv2/core.hpp;F:/opencv/sources/modules/core/include/opencv2/core/affine.hpp;F:/opencv/sources/modules/core/include/opencv2/core/base.hpp;F:/opencv/sources/modules/core/include/opencv2/core/bufferpool.hpp;F:/opencv/sources/modules/core/include/opencv2/core/core.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda.inl.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda_stream_accessor.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cuda_types.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cvstd.hpp;F:/opencv/sources/modules/core/include/opencv2/core/cvstd.inl.hpp;F:/opencv/sources/modules/core/include/opencv2/core/directx.hpp;F:/opencv/sources/modules/core/include/opencv2/core/eigen.hpp;F:/opencv/sources/modules/core/include/opencv2/core/fast_math.hpp;F:/opencv/sources/modules/core/include/opencv2/core/ippasync.hpp;F:/opencv/sources/modules/core/include/opencv2/core/mat.hpp;F:/opencv/sources/modules/core/include/opencv2/core/mat.inl.hpp;F:/opencv/sources/modules/core/include/opencv2/core/matx.hpp;F:/opencv/sources/modules/core/include/opencv2/core/neon_utils.hpp;F:/opencv/sources/modules/core/include/opencv2/core/ocl.hpp;F:/opencv/sources/modules/core/include/opencv2/core/ocl_genbase.hpp;F:/opencv/sources/modules/core/include/opencv2/core/opengl.hpp;F:/opencv/sources/modules/core/include/opencv2/core/operations.hpp;F:/opencv/sources/modules/core/include/opencv2/core/optim.hpp;F:/opencv/sources/modules/core/include/opencv2/core/ovx.hpp;F:/opencv/sources/modules/core/include/opencv2/core/persistence.hpp;F:/opencv/sources/modules/core/include/opencv2/core/private.cuda.hpp;F:/opencv/sources/modules/core/include/opencv2/core/private.hpp;F:/opencv/sources/modules/core/include/opencv2/core/ptr.inl.hpp;F:/opencv/sources/modules/core/include/opencv2/core/saturate.hpp;F:/opencv/sources/modules/core/include/opencv2/core/sse_utils.hpp;F:/opencv/sources/modules/core/include/opencv2/core/traits.hpp;F:/opencv/sources/modules/core/include/opencv2/core/types.hpp;F:/opencv/sources/modules/core/include/opencv2/core/utility.hpp;F:/opencv/sources/modules/core/include/opencv2/core/va_intel.hpp;F:/opencv/sources/modules/core/include/opencv2/core/version.hpp;F:/opencv/sources/modules/core/include/opencv2/core/wimage.hpp;F:/opencv/sources/modules/core/include/opencv2/core/core_c.h;F:/opencv/sources/modules/core/include/opencv2/core/cvdef.h;F:/opencv/sources/modules/core/include/opencv2/core/types_c.h;F:/opencv/sources/modules/core/include/opencv2/core/hal/hal.hpp;F:/opencv/sources/modules/core/include/opencv2/core/hal/intrin.hpp;F:/opencv/sources/modules/core/include/opencv2/core/hal/intrin_cpp.hpp;F:/opencv/sources/modules/core/include/opencv2/core/hal/intrin_neon.hpp;F:/opencv/sources/modules/core/include/opencv2/core/hal/intrin_sse.hpp;F:/opencv/sources/modules/core/include/opencv2/core/hal/interface.h
OPENCV_MODULE_opencv_core_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_core_LABEL:INTERNAL=Main;opencv_core
OPENCV_MODULE_opencv_core_LINK_DEPS:INTERNAL=;
//Location of opencv_core module sources
OPENCV_MODULE_opencv_core_LOCATION:INTERNAL=F:/opencv/sources/modules/core
//Optional dependencies of opencv_core module
OPENCV_MODULE_opencv_core_OPT_DEPS:INTERNAL=opencv_cudev
//Optional private dependencies of opencv_core module
OPENCV_MODULE_opencv_core_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_core module
OPENCV_MODULE_opencv_core_PRIVATE_REQ_DEPS:INTERNAL=zlib
//Required dependencies of opencv_core module
OPENCV_MODULE_opencv_core_REQ_DEPS:INTERNAL=
//List of source files for opencv_core
OPENCV_MODULE_opencv_core_SOURCES:INTERNAL=F:/opencv/sources/opencv3/modules/core/version_string.inc;F:/opencv/sources/modules/core/src/algorithm.cpp;F:/opencv/sources/modules/core/src/alloc.cpp;F:/opencv/sources/modules/core/src/arithm.cpp;F:/opencv/sources/modules/core/src/array.cpp;F:/opencv/sources/modules/core/src/command_line_parser.cpp;F:/opencv/sources/modules/core/src/conjugate_gradient.cpp;F:/opencv/sources/modules/core/src/convert.cpp;F:/opencv/sources/modules/core/src/copy.cpp;F:/opencv/sources/modules/core/src/cuda_gpu_mat.cpp;F:/opencv/sources/modules/core/src/cuda_host_mem.cpp;F:/opencv/sources/modules/core/src/cuda_info.cpp;F:/opencv/sources/modules/core/src/cuda_stream.cpp;F:/opencv/sources/modules/core/src/datastructs.cpp;F:/opencv/sources/modules/core/src/directx.cpp;F:/opencv/sources/modules/core/src/downhill_simplex.cpp;F:/opencv/sources/modules/core/src/dxt.cpp;F:/opencv/sources/modules/core/src/gl_core_3_1.cpp;F:/opencv/sources/modules/core/src/glob.cpp;F:/opencv/sources/modules/core/src/hal_internal.cpp;F:/opencv/sources/modules/core/src/kmeans.cpp;F:/opencv/sources/modules/core/src/lapack.cpp;F:/opencv/sources/modules/core/src/lda.cpp;F:/opencv/sources/modules/core/src/lpsolver.cpp;F:/opencv/sources/modules/core/src/mathfuncs.cpp;F:/opencv/sources/modules/core/src/mathfuncs_core.cpp;F:/opencv/sources/modules/core/src/matmul.cpp;F:/opencv/sources/modules/core/src/matop.cpp;F:/opencv/sources/modules/core/src/matrix.cpp;F:/opencv/sources/modules/core/src/matrix_decomp.cpp;F:/opencv/sources/modules/core/src/merge.cpp;F:/opencv/sources/modules/core/src/ocl.cpp;F:/opencv/sources/modules/core/src/opencl/runtime/opencl_clamdblas.cpp;F:/opencv/sources/modules/core/src/opencl/runtime/opencl_clamdfft.cpp;F:/opencv/sources/modules/core/src/opencl/runtime/opencl_core.cpp;F:/opencv/sources/modules/core/src/opengl.cpp;F:/opencv/sources/modules/core/src/out.cpp;F:/opencv/sources/modules/core/src/ovx.cpp;F:/opencv/sources/modules/core/src/parallel.cpp;F:/opencv/sources/modules/core/src/parallel_pthreads.cpp;F:/opencv/sources/modules/core/src/pca.cpp;F:/opencv/sources/modules/core/src/persistence.cpp;F:/opencv/sources/modules/core/src/rand.cpp;F:/opencv/sources/modules/core/src/split.cpp;F:/opencv/sources/modules/core/src/stat.cpp;F:/opencv/sources/modules/core/src/stl.cpp;F:/opencv/sources/modules/core/src/system.cpp;F:/opencv/sources/modules/core/src/tables.cpp;F:/opencv/sources/modules/core/src/types.cpp;F:/opencv/sources/modules/core/src/umatrix.cpp;F:/opencv/sources/modules/core/src/va_intel.cpp;F:/opencv/sources/modules/core/src/opencl/arithm.cl;F:/opencv/sources/modules/core/src/opencl/convert.cl;F:/opencv/sources/modules/core/src/opencl/copymakeborder.cl;F:/opencv/sources/modules/core/src/opencl/copyset.cl;F:/opencv/sources/modules/core/src/opencl/cvtclr_dx.cl;F:/opencv/sources/modules/core/src/opencl/fft.cl;F:/opencv/sources/modules/core/src/opencl/flip.cl;F:/opencv/sources/modules/core/src/opencl/gemm.cl;F:/opencv/sources/modules/core/src/opencl/inrange.cl;F:/opencv/sources/modules/core/src/opencl/lut.cl;F:/opencv/sources/modules/core/src/opencl/meanstddev.cl;F:/opencv/sources/modules/core/src/opencl/minmaxloc.cl;F:/opencv/sources/modules/core/src/opencl/mixchannels.cl;F:/opencv/sources/modules/core/src/opencl/mulspectrums.cl;F:/opencv/sources/modules/core/src/opencl/normalize.cl;F:/opencv/sources/modules/core/src/opencl/reduce.cl;F:/opencv/sources/modules/core/src/opencl/reduce2.cl;F:/opencv/sources/modules/core/src/opencl/repeat.cl;F:/opencv/sources/modules/core/src/opencl/set_identity.cl;F:/opencv/sources/modules/core/src/opencl/split_merge.cl;F:/opencv/sources/modules/core/src/opencl/transpose.cl;F:/opencv/sources/opencv3/modules/core/opencl_kernels_core.cpp;F:/opencv/sources/opencv3/modules/core/opencl_kernels_core.hpp;F:/opencv/sources/modules/core/src/arithm_core.hpp;F:/opencv/sources/modules/core/src/arithm_simd.hpp;F:/opencv/sources/modules/core/src/bufferpool.impl.hpp;F:/opencv/sources/modules/core/src/directx.inc.hpp;F:/opencv/sources/modules/core/src/gl_core_3_1.hpp;F:/opencv/sources/modules/core/src/hal_internal.hpp;F:/opencv/sources/modules/core/src/hal_replacement.hpp;F:/opencv/sources/modules/core/src/opencl/runtime/autogenerated/opencl_clamdblas_impl.hpp;F:/opencv/sources/modules/core/src/opencl/runtime/autogenerated/opencl_clamdfft_impl.hpp;F:/opencv/sources/modules/core/src/opencl/runtime/autogenerated/opencl_core_impl.hpp;F:/opencv/sources/modules/core/src/opencl/runtime/autogenerated/opencl_gl_impl.hpp;F:/opencv/sources/modules/core/src/opencl/runtime/runtime_common.hpp;F:/opencv/sources/modules/core/src/precomp.hpp
//List of wrappers supporting module opencv_core
OPENCV_MODULE_opencv_core_WRAPPERS:INTERNAL=java;python2;python3
//Location of opencv_cudaarithm module sources
OPENCV_MODULE_opencv_cudaarithm_LOCATION:INTERNAL=F:/opencv/sources/modules/cudaarithm
//Location of opencv_cudabgsegm module sources
OPENCV_MODULE_opencv_cudabgsegm_LOCATION:INTERNAL=F:/opencv/sources/modules/cudabgsegm
//Location of opencv_cudacodec module sources
OPENCV_MODULE_opencv_cudacodec_LOCATION:INTERNAL=F:/opencv/sources/modules/cudacodec
//Location of opencv_cudafeatures2d module sources
OPENCV_MODULE_opencv_cudafeatures2d_LOCATION:INTERNAL=F:/opencv/sources/modules/cudafeatures2d
//Location of opencv_cudafilters module sources
OPENCV_MODULE_opencv_cudafilters_LOCATION:INTERNAL=F:/opencv/sources/modules/cudafilters
//Location of opencv_cudaimgproc module sources
OPENCV_MODULE_opencv_cudaimgproc_LOCATION:INTERNAL=F:/opencv/sources/modules/cudaimgproc
//Location of opencv_cudalegacy module sources
OPENCV_MODULE_opencv_cudalegacy_LOCATION:INTERNAL=F:/opencv/sources/modules/cudalegacy
//Location of opencv_cudaobjdetect module sources
OPENCV_MODULE_opencv_cudaobjdetect_LOCATION:INTERNAL=F:/opencv/sources/modules/cudaobjdetect
//Location of opencv_cudaoptflow module sources
OPENCV_MODULE_opencv_cudaoptflow_LOCATION:INTERNAL=F:/opencv/sources/modules/cudaoptflow
//Location of opencv_cudastereo module sources
OPENCV_MODULE_opencv_cudastereo_LOCATION:INTERNAL=F:/opencv/sources/modules/cudastereo
//Location of opencv_cudawarping module sources
OPENCV_MODULE_opencv_cudawarping_LOCATION:INTERNAL=F:/opencv/sources/modules/cudawarping
//Location of opencv_cudev module sources
OPENCV_MODULE_opencv_cudev_LOCATION:INTERNAL=F:/opencv/sources/modules/cudev
OPENCV_MODULE_opencv_features2d_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/features2d
//List of opencv_features2d submodules
OPENCV_MODULE_opencv_features2d_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_features2d_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_features2d module
OPENCV_MODULE_opencv_features2d_DEPS:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_imgcodecs;opencv_videoio;opencv_highgui
//Extra dependencies of opencv_features2d module
OPENCV_MODULE_opencv_features2d_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_features2d module (for linker)
OPENCV_MODULE_opencv_features2d_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_imgcodecs;opencv_videoio;opencv_highgui
//Brief description of opencv_features2d module
OPENCV_MODULE_opencv_features2d_DESCRIPTION:INTERNAL=2D Features Framework
//List of header files for opencv_features2d
OPENCV_MODULE_opencv_features2d_HEADERS:INTERNAL=F:/opencv/sources/modules/features2d/include/opencv2/features2d.hpp;F:/opencv/sources/modules/features2d/include/opencv2/features2d/features2d.hpp
OPENCV_MODULE_opencv_features2d_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_features2d_LABEL:INTERNAL=Main;opencv_features2d
OPENCV_MODULE_opencv_features2d_LINK_DEPS:INTERNAL=;
//Location of opencv_features2d module sources
OPENCV_MODULE_opencv_features2d_LOCATION:INTERNAL=F:/opencv/sources/modules/features2d
//Optional dependencies of opencv_features2d module
OPENCV_MODULE_opencv_features2d_OPT_DEPS:INTERNAL=opencv_highgui
//Optional private dependencies of opencv_features2d module
OPENCV_MODULE_opencv_features2d_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_features2d module
OPENCV_MODULE_opencv_features2d_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_features2d module
OPENCV_MODULE_opencv_features2d_REQ_DEPS:INTERNAL=opencv_imgproc;opencv_ml;opencv_flann
//List of source files for opencv_features2d
OPENCV_MODULE_opencv_features2d_SOURCES:INTERNAL=F:/opencv/sources/modules/features2d/src/agast.cpp;F:/opencv/sources/modules/features2d/src/agast_score.cpp;F:/opencv/sources/modules/features2d/src/akaze.cpp;F:/opencv/sources/modules/features2d/src/bagofwords.cpp;F:/opencv/sources/modules/features2d/src/blobdetector.cpp;F:/opencv/sources/modules/features2d/src/brisk.cpp;F:/opencv/sources/modules/features2d/src/draw.cpp;F:/opencv/sources/modules/features2d/src/dynamic.cpp;F:/opencv/sources/modules/features2d/src/evaluation.cpp;F:/opencv/sources/modules/features2d/src/fast.cpp;F:/opencv/sources/modules/features2d/src/fast_score.cpp;F:/opencv/sources/modules/features2d/src/feature2d.cpp;F:/opencv/sources/modules/features2d/src/gftt.cpp;F:/opencv/sources/modules/features2d/src/kaze.cpp;F:/opencv/sources/modules/features2d/src/kaze/AKAZEFeatures.cpp;F:/opencv/sources/modules/features2d/src/kaze/KAZEFeatures.cpp;F:/opencv/sources/modules/features2d/src/kaze/fed.cpp;F:/opencv/sources/modules/features2d/src/kaze/nldiffusion_functions.cpp;F:/opencv/sources/modules/features2d/src/keypoint.cpp;F:/opencv/sources/modules/features2d/src/main.cpp;F:/opencv/sources/modules/features2d/src/matchers.cpp;F:/opencv/sources/modules/features2d/src/mser.cpp;F:/opencv/sources/modules/features2d/src/orb.cpp;F:/opencv/sources/modules/features2d/src/opencl/brute_force_match.cl;F:/opencv/sources/modules/features2d/src/opencl/fast.cl;F:/opencv/sources/modules/features2d/src/opencl/orb.cl;F:/opencv/sources/opencv3/modules/features2d/opencl_kernels_features2d.cpp;F:/opencv/sources/opencv3/modules/features2d/opencl_kernels_features2d.hpp;F:/opencv/sources/modules/features2d/src/agast_score.hpp;F:/opencv/sources/modules/features2d/src/fast_score.hpp;F:/opencv/sources/modules/features2d/src/precomp.hpp;F:/opencv/sources/modules/features2d/src/kaze/AKAZEConfig.h;F:/opencv/sources/modules/features2d/src/kaze/AKAZEFeatures.h;F:/opencv/sources/modules/features2d/src/kaze/KAZEConfig.h;F:/opencv/sources/modules/features2d/src/kaze/KAZEFeatures.h;F:/opencv/sources/modules/features2d/src/kaze/TEvolution.h;F:/opencv/sources/modules/features2d/src/kaze/fed.h;F:/opencv/sources/modules/features2d/src/kaze/nldiffusion_functions.h;F:/opencv/sources/modules/features2d/src/kaze/utils.h
//List of wrappers supporting module opencv_features2d
OPENCV_MODULE_opencv_features2d_WRAPPERS:INTERNAL=java;python2;python3
OPENCV_MODULE_opencv_flann_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/flann
//List of opencv_flann submodules
OPENCV_MODULE_opencv_flann_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_flann_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_flann module
OPENCV_MODULE_opencv_flann_DEPS:INTERNAL=opencv_core
//Extra dependencies of opencv_flann module
OPENCV_MODULE_opencv_flann_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_flann module (for linker)
OPENCV_MODULE_opencv_flann_DEPS_TO_LINK:INTERNAL=opencv_core
//Brief description of opencv_flann module
OPENCV_MODULE_opencv_flann_DESCRIPTION:INTERNAL=Clustering and Search in Multi-Dimensional Spaces
//List of header files for opencv_flann
OPENCV_MODULE_opencv_flann_HEADERS:INTERNAL=F:/opencv/sources/modules/flann/include/opencv2/flann.hpp;F:/opencv/sources/modules/flann/include/opencv2/flann/flann.hpp;F:/opencv/sources/modules/flann/include/opencv2/flann/flann_base.hpp;F:/opencv/sources/modules/flann/include/opencv2/flann/miniflann.hpp;F:/opencv/sources/modules/flann/include/opencv2/flann/all_indices.h;F:/opencv/sources/modules/flann/include/opencv2/flann/allocator.h;F:/opencv/sources/modules/flann/include/opencv2/flann/any.h;F:/opencv/sources/modules/flann/include/opencv2/flann/autotuned_index.h;F:/opencv/sources/modules/flann/include/opencv2/flann/composite_index.h;F:/opencv/sources/modules/flann/include/opencv2/flann/config.h;F:/opencv/sources/modules/flann/include/opencv2/flann/defines.h;F:/opencv/sources/modules/flann/include/opencv2/flann/dist.h;F:/opencv/sources/modules/flann/include/opencv2/flann/dummy.h;F:/opencv/sources/modules/flann/include/opencv2/flann/dynamic_bitset.h;F:/opencv/sources/modules/flann/include/opencv2/flann/general.h;F:/opencv/sources/modules/flann/include/opencv2/flann/ground_truth.h;F:/opencv/sources/modules/flann/include/opencv2/flann/hdf5.h;F:/opencv/sources/modules/flann/include/opencv2/flann/heap.h;F:/opencv/sources/modules/flann/include/opencv2/flann/hierarchical_clustering_index.h;F:/opencv/sources/modules/flann/include/opencv2/flann/index_testing.h;F:/opencv/sources/modules/flann/include/opencv2/flann/kdtree_index.h;F:/opencv/sources/modules/flann/include/opencv2/flann/kdtree_single_index.h;F:/opencv/sources/modules/flann/include/opencv2/flann/kmeans_index.h;F:/opencv/sources/modules/flann/include/opencv2/flann/linear_index.h;F:/opencv/sources/modules/flann/include/opencv2/flann/logger.h;F:/opencv/sources/modules/flann/include/opencv2/flann/lsh_index.h;F:/opencv/sources/modules/flann/include/opencv2/flann/lsh_table.h;F:/opencv/sources/modules/flann/include/opencv2/flann/matrix.h;F:/opencv/sources/modules/flann/include/opencv2/flann/nn_index.h;F:/opencv/sources/modules/flann/include/opencv2/flann/object_factory.h;F:/opencv/sources/modules/flann/include/opencv2/flann/params.h;F:/opencv/sources/modules/flann/include/opencv2/flann/random.h;F:/opencv/sources/modules/flann/include/opencv2/flann/result_set.h;F:/opencv/sources/modules/flann/include/opencv2/flann/sampling.h;F:/opencv/sources/modules/flann/include/opencv2/flann/saving.h;F:/opencv/sources/modules/flann/include/opencv2/flann/simplex_downhill.h;F:/opencv/sources/modules/flann/include/opencv2/flann/timer.h
OPENCV_MODULE_opencv_flann_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_flann_LABEL:INTERNAL=Main;opencv_flann
OPENCV_MODULE_opencv_flann_LINK_DEPS:INTERNAL=;
//Location of opencv_flann module sources
OPENCV_MODULE_opencv_flann_LOCATION:INTERNAL=F:/opencv/sources/modules/flann
//Optional dependencies of opencv_flann module
OPENCV_MODULE_opencv_flann_OPT_DEPS:INTERNAL=
//Optional private dependencies of opencv_flann module
OPENCV_MODULE_opencv_flann_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_flann module
OPENCV_MODULE_opencv_flann_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_flann module
OPENCV_MODULE_opencv_flann_REQ_DEPS:INTERNAL=opencv_core
//List of source files for opencv_flann
OPENCV_MODULE_opencv_flann_SOURCES:INTERNAL=F:/opencv/sources/modules/flann/src/flann.cpp;F:/opencv/sources/modules/flann/src/miniflann.cpp;F:/opencv/sources/modules/flann/src/precomp.hpp
//List of wrappers supporting module opencv_flann
OPENCV_MODULE_opencv_flann_WRAPPERS:INTERNAL=python2;python3
OPENCV_MODULE_opencv_highgui_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/highgui
//List of opencv_highgui submodules
OPENCV_MODULE_opencv_highgui_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_highgui_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_highgui module
OPENCV_MODULE_opencv_highgui_DEPS:INTERNAL=opencv_core;opencv_imgproc;opencv_imgcodecs;opencv_videoio
//Extra dependencies of opencv_highgui module
OPENCV_MODULE_opencv_highgui_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_highgui module (for linker)
OPENCV_MODULE_opencv_highgui_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_imgproc;opencv_imgcodecs;opencv_videoio
//Brief description of opencv_highgui module
OPENCV_MODULE_opencv_highgui_DESCRIPTION:INTERNAL=High-level GUI and Media I/O
//List of header files for opencv_highgui
OPENCV_MODULE_opencv_highgui_HEADERS:INTERNAL=F:/opencv/sources/modules/highgui/include/opencv2/highgui.hpp;F:/opencv/sources/modules/highgui/include/opencv2/highgui/highgui.hpp;F:/opencv/sources/modules/highgui/include/opencv2/highgui/highgui_c.h
OPENCV_MODULE_opencv_highgui_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_highgui_LABEL:INTERNAL=Main;opencv_highgui
OPENCV_MODULE_opencv_highgui_LINK_DEPS:INTERNAL=;comctl32;gdi32;ole32;setupapi;ws2_32
//Location of opencv_highgui module sources
OPENCV_MODULE_opencv_highgui_LOCATION:INTERNAL=F:/opencv/sources/modules/highgui
//Optional dependencies of opencv_highgui module
OPENCV_MODULE_opencv_highgui_OPT_DEPS:INTERNAL=opencv_imgcodecs;opencv_videoio
//Optional private dependencies of opencv_highgui module
OPENCV_MODULE_opencv_highgui_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_highgui module
OPENCV_MODULE_opencv_highgui_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_highgui module
OPENCV_MODULE_opencv_highgui_REQ_DEPS:INTERNAL=opencv_imgproc
//List of source files for opencv_highgui
OPENCV_MODULE_opencv_highgui_SOURCES:INTERNAL=F:/opencv/sources/modules/highgui/src/window.cpp;F:/opencv/sources/modules/highgui/src/window_w32.cpp;F:/opencv/sources/modules/highgui/src/precomp.hpp
//List of wrappers supporting module opencv_highgui
OPENCV_MODULE_opencv_highgui_WRAPPERS:INTERNAL=python2;python3
OPENCV_MODULE_opencv_imgcodecs_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/imgcodecs
//List of opencv_imgcodecs submodules
OPENCV_MODULE_opencv_imgcodecs_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_imgcodecs_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_imgcodecs module
OPENCV_MODULE_opencv_imgcodecs_DEPS:INTERNAL=opencv_core;opencv_imgproc
//Extra dependencies of opencv_imgcodecs module
OPENCV_MODULE_opencv_imgcodecs_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_imgcodecs module (for linker)
OPENCV_MODULE_opencv_imgcodecs_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_imgproc
//Brief description of opencv_imgcodecs module
OPENCV_MODULE_opencv_imgcodecs_DESCRIPTION:INTERNAL=Image codecs
//List of header files for opencv_imgcodecs
OPENCV_MODULE_opencv_imgcodecs_HEADERS:INTERNAL=F:/opencv/sources/modules/imgcodecs/include/opencv2/imgcodecs.hpp;F:/opencv/sources/modules/imgcodecs/include/opencv2/imgcodecs/imgcodecs.hpp;F:/opencv/sources/modules/imgcodecs/include/opencv2/imgcodecs/imgcodecs_c.h;F:/opencv/sources/modules/imgcodecs/include/opencv2/imgcodecs/ios.h
OPENCV_MODULE_opencv_imgcodecs_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_imgcodecs_LABEL:INTERNAL=Main;opencv_imgcodecs
OPENCV_MODULE_opencv_imgcodecs_LINK_DEPS:INTERNAL=;zlib;libjpeg;libwebp;libpng;libtiff;libjasper;IlmImf
//Location of opencv_imgcodecs module sources
OPENCV_MODULE_opencv_imgcodecs_LOCATION:INTERNAL=F:/opencv/sources/modules/imgcodecs
//Optional dependencies of opencv_imgcodecs module
OPENCV_MODULE_opencv_imgcodecs_OPT_DEPS:INTERNAL=
//Optional private dependencies of opencv_imgcodecs module
OPENCV_MODULE_opencv_imgcodecs_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_imgcodecs module
OPENCV_MODULE_opencv_imgcodecs_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_imgcodecs module
OPENCV_MODULE_opencv_imgcodecs_REQ_DEPS:INTERNAL=opencv_imgproc
//List of source files for opencv_imgcodecs
OPENCV_MODULE_opencv_imgcodecs_SOURCES:INTERNAL=F:/opencv/sources/modules/imgcodecs/src/loadsave.cpp;F:/opencv/sources/modules/imgcodecs/src/utils.cpp;F:/opencv/sources/modules/imgcodecs/src/precomp.hpp;F:/opencv/sources/modules/imgcodecs/src/utils.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_base.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_bmp.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_exr.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_gdal.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_gdcm.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_hdr.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_jpeg.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_jpeg2000.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_pam.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_png.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_pxm.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_sunras.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_tiff.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_webp.cpp;F:/opencv/sources/modules/imgcodecs/src/bitstrm.cpp;F:/opencv/sources/modules/imgcodecs/src/rgbe.cpp;F:/opencv/sources/modules/imgcodecs/src/exif.cpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_base.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_bmp.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_exr.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_gdal.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_gdcm.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_hdr.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_jpeg.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_jpeg2000.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_pam.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_png.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_pxm.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_sunras.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_tiff.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmt_webp.hpp;F:/opencv/sources/modules/imgcodecs/src/grfmts.hpp;F:/opencv/sources/modules/imgcodecs/src/bitstrm.hpp;F:/opencv/sources/modules/imgcodecs/src/rgbe.hpp;F:/opencv/sources/modules/imgcodecs/src/exif.hpp
//List of wrappers supporting module opencv_imgcodecs
OPENCV_MODULE_opencv_imgcodecs_WRAPPERS:INTERNAL=java;python2;python3
OPENCV_MODULE_opencv_imgproc_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/imgproc
//List of opencv_imgproc submodules
OPENCV_MODULE_opencv_imgproc_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_imgproc_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_imgproc module
OPENCV_MODULE_opencv_imgproc_DEPS:INTERNAL=opencv_core
//Extra dependencies of opencv_imgproc module
OPENCV_MODULE_opencv_imgproc_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_imgproc module (for linker)
OPENCV_MODULE_opencv_imgproc_DEPS_TO_LINK:INTERNAL=opencv_core
//Brief description of opencv_imgproc module
OPENCV_MODULE_opencv_imgproc_DESCRIPTION:INTERNAL=Image Processing
//List of header files for opencv_imgproc
OPENCV_MODULE_opencv_imgproc_HEADERS:INTERNAL=F:/opencv/sources/modules/imgproc/include/opencv2/imgproc.hpp;F:/opencv/sources/modules/imgproc/include/opencv2/imgproc/imgproc.hpp;F:/opencv/sources/modules/imgproc/include/opencv2/imgproc/imgproc_c.h;F:/opencv/sources/modules/imgproc/include/opencv2/imgproc/types_c.h;F:/opencv/sources/modules/imgproc/include/opencv2/imgproc/hal/hal.hpp;F:/opencv/sources/modules/imgproc/include/opencv2/imgproc/hal/interface.h;F:/opencv/sources/modules/imgproc/include/opencv2/imgproc/detail/distortion_model.hpp
OPENCV_MODULE_opencv_imgproc_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_imgproc_LABEL:INTERNAL=Main;opencv_imgproc
OPENCV_MODULE_opencv_imgproc_LINK_DEPS:INTERNAL=;
//Location of opencv_imgproc module sources
OPENCV_MODULE_opencv_imgproc_LOCATION:INTERNAL=F:/opencv/sources/modules/imgproc
//Optional dependencies of opencv_imgproc module
OPENCV_MODULE_opencv_imgproc_OPT_DEPS:INTERNAL=
//Optional private dependencies of opencv_imgproc module
OPENCV_MODULE_opencv_imgproc_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_imgproc module
OPENCV_MODULE_opencv_imgproc_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_imgproc module
OPENCV_MODULE_opencv_imgproc_REQ_DEPS:INTERNAL=opencv_core
//List of source files for opencv_imgproc
OPENCV_MODULE_opencv_imgproc_SOURCES:INTERNAL=F:/opencv/sources/modules/imgproc/src/accum.cpp;F:/opencv/sources/modules/imgproc/src/approx.cpp;F:/opencv/sources/modules/imgproc/src/blend.cpp;F:/opencv/sources/modules/imgproc/src/canny.cpp;F:/opencv/sources/modules/imgproc/src/clahe.cpp;F:/opencv/sources/modules/imgproc/src/color.cpp;F:/opencv/sources/modules/imgproc/src/colormap.cpp;F:/opencv/sources/modules/imgproc/src/connectedcomponents.cpp;F:/opencv/sources/modules/imgproc/src/contours.cpp;F:/opencv/sources/modules/imgproc/src/convhull.cpp;F:/opencv/sources/modules/imgproc/src/corner.cpp;F:/opencv/sources/modules/imgproc/src/cornersubpix.cpp;F:/opencv/sources/modules/imgproc/src/demosaicing.cpp;F:/opencv/sources/modules/imgproc/src/deriv.cpp;F:/opencv/sources/modules/imgproc/src/distransform.cpp;F:/opencv/sources/modules/imgproc/src/drawing.cpp;F:/opencv/sources/modules/imgproc/src/emd.cpp;F:/opencv/sources/modules/imgproc/src/featureselect.cpp;F:/opencv/sources/modules/imgproc/src/filter.cpp;F:/opencv/sources/modules/imgproc/src/floodfill.cpp;F:/opencv/sources/modules/imgproc/src/gabor.cpp;F:/opencv/sources/modules/imgproc/src/generalized_hough.cpp;F:/opencv/sources/modules/imgproc/src/geometry.cpp;F:/opencv/sources/modules/imgproc/src/grabcut.cpp;F:/opencv/sources/modules/imgproc/src/hershey_fonts.cpp;F:/opencv/sources/modules/imgproc/src/histogram.cpp;F:/opencv/sources/modules/imgproc/src/hough.cpp;F:/opencv/sources/modules/imgproc/src/imgwarp.cpp;F:/opencv/sources/modules/imgproc/src/intersection.cpp;F:/opencv/sources/modules/imgproc/src/linefit.cpp;F:/opencv/sources/modules/imgproc/src/lsd.cpp;F:/opencv/sources/modules/imgproc/src/main.cpp;F:/opencv/sources/modules/imgproc/src/matchcontours.cpp;F:/opencv/sources/modules/imgproc/src/min_enclosing_triangle.cpp;F:/opencv/sources/modules/imgproc/src/moments.cpp;F:/opencv/sources/modules/imgproc/src/morph.cpp;F:/opencv/sources/modules/imgproc/src/phasecorr.cpp;F:/opencv/sources/modules/imgproc/src/pyramids.cpp;F:/opencv/sources/modules/imgproc/src/rotcalipers.cpp;F:/opencv/sources/modules/imgproc/src/samplers.cpp;F:/opencv/sources/modules/imgproc/src/segmentation.cpp;F:/opencv/sources/modules/imgproc/src/shapedescr.cpp;F:/opencv/sources/modules/imgproc/src/smooth.cpp;F:/opencv/sources/modules/imgproc/src/spatialgradient.cpp;F:/opencv/sources/modules/imgproc/src/subdivision2d.cpp;F:/opencv/sources/modules/imgproc/src/sumpixels.cpp;F:/opencv/sources/modules/imgproc/src/tables.cpp;F:/opencv/sources/modules/imgproc/src/templmatch.cpp;F:/opencv/sources/modules/imgproc/src/thresh.cpp;F:/opencv/sources/modules/imgproc/src/undistort.cpp;F:/opencv/sources/modules/imgproc/src/utils.cpp;F:/opencv/sources/modules/imgproc/src/opencl/accumulate.cl;F:/opencv/sources/modules/imgproc/src/opencl/bilateral.cl;F:/opencv/sources/modules/imgproc/src/opencl/blend_linear.cl;F:/opencv/sources/modules/imgproc/src/opencl/boxFilter.cl;F:/opencv/sources/modules/imgproc/src/opencl/boxFilter3x3.cl;F:/opencv/sources/modules/imgproc/src/opencl/calc_back_project.cl;F:/opencv/sources/modules/imgproc/src/opencl/canny.cl;F:/opencv/sources/modules/imgproc/src/opencl/clahe.cl;F:/opencv/sources/modules/imgproc/src/opencl/corner.cl;F:/opencv/sources/modules/imgproc/src/opencl/covardata.cl;F:/opencv/sources/modules/imgproc/src/opencl/cvtcolor.cl;F:/opencv/sources/modules/imgproc/src/opencl/filter2D.cl;F:/opencv/sources/modules/imgproc/src/opencl/filter2DSmall.cl;F:/opencv/sources/modules/imgproc/src/opencl/filterSepCol.cl;F:/opencv/sources/modules/imgproc/src/opencl/filterSepRow.cl;F:/opencv/sources/modules/imgproc/src/opencl/filterSep_singlePass.cl;F:/opencv/sources/modules/imgproc/src/opencl/filterSmall.cl;F:/opencv/sources/modules/imgproc/src/opencl/gaussianBlur3x3.cl;F:/opencv/sources/modules/imgproc/src/opencl/gaussianBlur5x5.cl;F:/opencv/sources/modules/imgproc/src/opencl/gftt.cl;F:/opencv/sources/modules/imgproc/src/opencl/histogram.cl;F:/opencv/sources/modules/imgproc/src/opencl/hough_lines.cl;F:/opencv/sources/modules/imgproc/src/opencl/integral_sum.cl;F:/opencv/sources/modules/imgproc/src/opencl/laplacian3.cl;F:/opencv/sources/modules/imgproc/src/opencl/laplacian5.cl;F:/opencv/sources/modules/imgproc/src/opencl/linearPolar.cl;F:/opencv/sources/modules/imgproc/src/opencl/logPolar.cl;F:/opencv/sources/modules/imgproc/src/opencl/match_template.cl;F:/opencv/sources/modules/imgproc/src/opencl/medianFilter.cl;F:/opencv/sources/modules/imgproc/src/opencl/moments.cl;F:/opencv/sources/modules/imgproc/src/opencl/morph.cl;F:/opencv/sources/modules/imgproc/src/opencl/morph3x3.cl;F:/opencv/sources/modules/imgproc/src/opencl/precornerdetect.cl;F:/opencv/sources/modules/imgproc/src/opencl/pyr_down.cl;F:/opencv/sources/modules/imgproc/src/opencl/pyr_up.cl;F:/opencv/sources/modules/imgproc/src/opencl/pyramid_up.cl;F:/opencv/sources/modules/imgproc/src/opencl/remap.cl;F:/opencv/sources/modules/imgproc/src/opencl/resize.cl;F:/opencv/sources/modules/imgproc/src/opencl/sepFilter3x3.cl;F:/opencv/sources/modules/imgproc/src/opencl/threshold.cl;F:/opencv/sources/modules/imgproc/src/opencl/warp_affine.cl;F:/opencv/sources/modules/imgproc/src/opencl/warp_perspective.cl;F:/opencv/sources/modules/imgproc/src/opencl/warp_transform.cl;F:/opencv/sources/opencv3/modules/imgproc/opencl_kernels_imgproc.cpp;F:/opencv/sources/opencv3/modules/imgproc/opencl_kernels_imgproc.hpp;F:/opencv/sources/modules/imgproc/src/filterengine.hpp;F:/opencv/sources/modules/imgproc/src/gcgraph.hpp;F:/opencv/sources/modules/imgproc/src/hal_replacement.hpp;F:/opencv/sources/modules/imgproc/src/precomp.hpp;F:/opencv/sources/modules/imgproc/src/_geom.h
//List of wrappers supporting module opencv_imgproc
OPENCV_MODULE_opencv_imgproc_WRAPPERS:INTERNAL=java;python2;python3
OPENCV_MODULE_opencv_java_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/java
//List of opencv_java submodules
OPENCV_MODULE_opencv_java_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_java_CLASS:INTERNAL=BINDINGS
//Flattened dependencies of opencv_java module
OPENCV_MODULE_opencv_java_DEPS:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_photo;opencv_video;opencv_imgcodecs;opencv_videoio;opencv_highgui;opencv_objdetect;opencv_features2d;opencv_calib3d
//Extra dependencies of opencv_java module
OPENCV_MODULE_opencv_java_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_java module (for linker)
OPENCV_MODULE_opencv_java_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_photo;opencv_video;opencv_imgcodecs;opencv_videoio;opencv_highgui;opencv_objdetect;opencv_features2d;opencv_calib3d
//Brief description of opencv_java module
OPENCV_MODULE_opencv_java_DESCRIPTION:INTERNAL=The java bindings
OPENCV_MODULE_opencv_java_IS_PART_OF_WORLD:INTERNAL=OFF
OPENCV_MODULE_opencv_java_LABEL:INTERNAL=Main;opencv_java
OPENCV_MODULE_opencv_java_LINK_DEPS:INTERNAL=
//Location of opencv_java module sources
OPENCV_MODULE_opencv_java_LOCATION:INTERNAL=F:/opencv/sources/modules/java
//Optional dependencies of opencv_java module
OPENCV_MODULE_opencv_java_OPT_DEPS:INTERNAL=opencv_calib3d;opencv_core;opencv_features2d;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_video;opencv_videoio
//Optional private dependencies of opencv_java module
OPENCV_MODULE_opencv_java_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_java module
OPENCV_MODULE_opencv_java_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_java module
OPENCV_MODULE_opencv_java_REQ_DEPS:INTERNAL=opencv_core;opencv_imgproc
//List of wrappers supporting module opencv_java
OPENCV_MODULE_opencv_java_WRAPPERS:INTERNAL=
OPENCV_MODULE_opencv_ml_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/ml
//List of opencv_ml submodules
OPENCV_MODULE_opencv_ml_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_ml_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_ml module
OPENCV_MODULE_opencv_ml_DEPS:INTERNAL=opencv_core
//Extra dependencies of opencv_ml module
OPENCV_MODULE_opencv_ml_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_ml module (for linker)
OPENCV_MODULE_opencv_ml_DEPS_TO_LINK:INTERNAL=opencv_core
//Brief description of opencv_ml module
OPENCV_MODULE_opencv_ml_DESCRIPTION:INTERNAL=Machine Learning
//List of header files for opencv_ml
OPENCV_MODULE_opencv_ml_HEADERS:INTERNAL=F:/opencv/sources/modules/ml/include/opencv2/ml.hpp;F:/opencv/sources/modules/ml/include/opencv2/ml/ml.hpp
OPENCV_MODULE_opencv_ml_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_ml_LABEL:INTERNAL=Main;opencv_ml
OPENCV_MODULE_opencv_ml_LINK_DEPS:INTERNAL=;
//Location of opencv_ml module sources
OPENCV_MODULE_opencv_ml_LOCATION:INTERNAL=F:/opencv/sources/modules/ml
//Optional dependencies of opencv_ml module
OPENCV_MODULE_opencv_ml_OPT_DEPS:INTERNAL=
//Optional private dependencies of opencv_ml module
OPENCV_MODULE_opencv_ml_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_ml module
OPENCV_MODULE_opencv_ml_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_ml module
OPENCV_MODULE_opencv_ml_REQ_DEPS:INTERNAL=opencv_core
//List of source files for opencv_ml
OPENCV_MODULE_opencv_ml_SOURCES:INTERNAL=F:/opencv/sources/modules/ml/src/ann_mlp.cpp;F:/opencv/sources/modules/ml/src/boost.cpp;F:/opencv/sources/modules/ml/src/data.cpp;F:/opencv/sources/modules/ml/src/em.cpp;F:/opencv/sources/modules/ml/src/gbt.cpp;F:/opencv/sources/modules/ml/src/inner_functions.cpp;F:/opencv/sources/modules/ml/src/kdtree.cpp;F:/opencv/sources/modules/ml/src/knearest.cpp;F:/opencv/sources/modules/ml/src/lr.cpp;F:/opencv/sources/modules/ml/src/nbayes.cpp;F:/opencv/sources/modules/ml/src/rtrees.cpp;F:/opencv/sources/modules/ml/src/svm.cpp;F:/opencv/sources/modules/ml/src/svmsgd.cpp;F:/opencv/sources/modules/ml/src/testset.cpp;F:/opencv/sources/modules/ml/src/tree.cpp;F:/opencv/sources/modules/ml/src/kdtree.hpp;F:/opencv/sources/modules/ml/src/precomp.hpp
//List of wrappers supporting module opencv_ml
OPENCV_MODULE_opencv_ml_WRAPPERS:INTERNAL=java;python2;python3
OPENCV_MODULE_opencv_objdetect_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/objdetect
//List of opencv_objdetect submodules
OPENCV_MODULE_opencv_objdetect_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_objdetect_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_objdetect module
OPENCV_MODULE_opencv_objdetect_DEPS:INTERNAL=opencv_core;opencv_imgproc;opencv_ml;opencv_imgcodecs;opencv_videoio;opencv_highgui
//Extra dependencies of opencv_objdetect module
OPENCV_MODULE_opencv_objdetect_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_objdetect module (for linker)
OPENCV_MODULE_opencv_objdetect_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_imgproc;opencv_ml;opencv_imgcodecs;opencv_videoio;opencv_highgui
//Brief description of opencv_objdetect module
OPENCV_MODULE_opencv_objdetect_DESCRIPTION:INTERNAL=Object Detection
//List of header files for opencv_objdetect
OPENCV_MODULE_opencv_objdetect_HEADERS:INTERNAL=F:/opencv/sources/modules/objdetect/include/opencv2/objdetect.hpp;F:/opencv/sources/modules/objdetect/include/opencv2/objdetect/detection_based_tracker.hpp;F:/opencv/sources/modules/objdetect/include/opencv2/objdetect/objdetect.hpp;F:/opencv/sources/modules/objdetect/include/opencv2/objdetect/objdetect_c.h
OPENCV_MODULE_opencv_objdetect_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_objdetect_LABEL:INTERNAL=Main;opencv_objdetect
OPENCV_MODULE_opencv_objdetect_LINK_DEPS:INTERNAL=;
//Location of opencv_objdetect module sources
OPENCV_MODULE_opencv_objdetect_LOCATION:INTERNAL=F:/opencv/sources/modules/objdetect
//Optional dependencies of opencv_objdetect module
OPENCV_MODULE_opencv_objdetect_OPT_DEPS:INTERNAL=opencv_highgui
//Optional private dependencies of opencv_objdetect module
OPENCV_MODULE_opencv_objdetect_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_objdetect module
OPENCV_MODULE_opencv_objdetect_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_objdetect module
OPENCV_MODULE_opencv_objdetect_REQ_DEPS:INTERNAL=opencv_core;opencv_imgproc;opencv_ml
//List of source files for opencv_objdetect
OPENCV_MODULE_opencv_objdetect_SOURCES:INTERNAL=F:/opencv/sources/modules/objdetect/src/cascadedetect.cpp;F:/opencv/sources/modules/objdetect/src/cascadedetect_convert.cpp;F:/opencv/sources/modules/objdetect/src/detection_based_tracker.cpp;F:/opencv/sources/modules/objdetect/src/haar.cpp;F:/opencv/sources/modules/objdetect/src/hog.cpp;F:/opencv/sources/modules/objdetect/src/main.cpp;F:/opencv/sources/modules/objdetect/src/opencl/cascadedetect.cl;F:/opencv/sources/modules/objdetect/src/opencl/objdetect_hog.cl;F:/opencv/sources/opencv3/modules/objdetect/opencl_kernels_objdetect.cpp;F:/opencv/sources/opencv3/modules/objdetect/opencl_kernels_objdetect.hpp;F:/opencv/sources/modules/objdetect/src/cascadedetect.hpp;F:/opencv/sources/modules/objdetect/src/precomp.hpp
//List of wrappers supporting module opencv_objdetect
OPENCV_MODULE_opencv_objdetect_WRAPPERS:INTERNAL=java;python2;python3
OPENCV_MODULE_opencv_photo_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/photo
//List of opencv_photo submodules
OPENCV_MODULE_opencv_photo_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_photo_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_photo module
OPENCV_MODULE_opencv_photo_DEPS:INTERNAL=opencv_core;opencv_imgproc
//Extra dependencies of opencv_photo module
OPENCV_MODULE_opencv_photo_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_photo module (for linker)
OPENCV_MODULE_opencv_photo_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_imgproc
//Brief description of opencv_photo module
OPENCV_MODULE_opencv_photo_DESCRIPTION:INTERNAL=Computational Photography
//List of header files for opencv_photo
OPENCV_MODULE_opencv_photo_HEADERS:INTERNAL=F:/opencv/sources/modules/photo/include/opencv2/photo.hpp;F:/opencv/sources/modules/photo/include/opencv2/photo/cuda.hpp;F:/opencv/sources/modules/photo/include/opencv2/photo/photo.hpp;F:/opencv/sources/modules/photo/include/opencv2/photo/photo_c.h
OPENCV_MODULE_opencv_photo_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_photo_LABEL:INTERNAL=Main;opencv_photo
OPENCV_MODULE_opencv_photo_LINK_DEPS:INTERNAL=;
//Location of opencv_photo module sources
OPENCV_MODULE_opencv_photo_LOCATION:INTERNAL=F:/opencv/sources/modules/photo
//Optional dependencies of opencv_photo module
OPENCV_MODULE_opencv_photo_OPT_DEPS:INTERNAL=opencv_cudaarithm;opencv_cudaimgproc
//Optional private dependencies of opencv_photo module
OPENCV_MODULE_opencv_photo_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_photo module
OPENCV_MODULE_opencv_photo_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_photo module
OPENCV_MODULE_opencv_photo_REQ_DEPS:INTERNAL=opencv_imgproc
//List of source files for opencv_photo
OPENCV_MODULE_opencv_photo_SOURCES:INTERNAL=F:/opencv/sources/modules/photo/src/align.cpp;F:/opencv/sources/modules/photo/src/calibrate.cpp;F:/opencv/sources/modules/photo/src/contrast_preserve.cpp;F:/opencv/sources/modules/photo/src/denoise_tvl1.cpp;F:/opencv/sources/modules/photo/src/denoising.cpp;F:/opencv/sources/modules/photo/src/denoising.cuda.cpp;F:/opencv/sources/modules/photo/src/hdr_common.cpp;F:/opencv/sources/modules/photo/src/inpaint.cpp;F:/opencv/sources/modules/photo/src/merge.cpp;F:/opencv/sources/modules/photo/src/npr.cpp;F:/opencv/sources/modules/photo/src/seamless_cloning.cpp;F:/opencv/sources/modules/photo/src/seamless_cloning_impl.cpp;F:/opencv/sources/modules/photo/src/tonemap.cpp;F:/opencv/sources/modules/photo/src/opencl/nlmeans.cl;F:/opencv/sources/opencv3/modules/photo/opencl_kernels_photo.cpp;F:/opencv/sources/opencv3/modules/photo/opencl_kernels_photo.hpp;F:/opencv/sources/modules/photo/src/arrays.hpp;F:/opencv/sources/modules/photo/src/contrast_preserve.hpp;F:/opencv/sources/modules/photo/src/fast_nlmeans_denoising_invoker.hpp;F:/opencv/sources/modules/photo/src/fast_nlmeans_denoising_invoker_commons.hpp;F:/opencv/sources/modules/photo/src/fast_nlmeans_denoising_opencl.hpp;F:/opencv/sources/modules/photo/src/fast_nlmeans_multi_denoising_invoker.hpp;F:/opencv/sources/modules/photo/src/hdr_common.hpp;F:/opencv/sources/modules/photo/src/npr.hpp;F:/opencv/sources/modules/photo/src/precomp.hpp;F:/opencv/sources/modules/photo/src/seamless_cloning.hpp
//List of wrappers supporting module opencv_photo
OPENCV_MODULE_opencv_photo_WRAPPERS:INTERNAL=java;python2;python3
//Location of opencv_python2 module sources
OPENCV_MODULE_opencv_python2_LOCATION:INTERNAL=F:/opencv/sources/modules/python/python2
//Optional dependencies of opencv_python2 module
OPENCV_MODULE_opencv_python2_OPT_DEPS:INTERNAL=opencv_calib3d;opencv_core;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab
//Optional private dependencies of opencv_python2 module
OPENCV_MODULE_opencv_python2_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_python2 module
OPENCV_MODULE_opencv_python2_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_python2 module
OPENCV_MODULE_opencv_python2_REQ_DEPS:INTERNAL=
//List of wrappers supporting module opencv_python2
OPENCV_MODULE_opencv_python2_WRAPPERS:INTERNAL=
//Location of opencv_python3 module sources
OPENCV_MODULE_opencv_python3_LOCATION:INTERNAL=F:/opencv/sources/modules/python/python3
//Optional dependencies of opencv_python3 module
OPENCV_MODULE_opencv_python3_OPT_DEPS:INTERNAL=opencv_calib3d;opencv_core;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab
//Optional private dependencies of opencv_python3 module
OPENCV_MODULE_opencv_python3_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_python3 module
OPENCV_MODULE_opencv_python3_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_python3 module
OPENCV_MODULE_opencv_python3_REQ_DEPS:INTERNAL=
//List of wrappers supporting module opencv_python3
OPENCV_MODULE_opencv_python3_WRAPPERS:INTERNAL=
OPENCV_MODULE_opencv_shape_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/shape
//List of opencv_shape submodules
OPENCV_MODULE_opencv_shape_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_shape_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_shape module
OPENCV_MODULE_opencv_shape_DEPS:INTERNAL=opencv_core;opencv_imgproc;opencv_video
//Extra dependencies of opencv_shape module
OPENCV_MODULE_opencv_shape_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_shape module (for linker)
OPENCV_MODULE_opencv_shape_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_imgproc;opencv_video
//Brief description of opencv_shape module
OPENCV_MODULE_opencv_shape_DESCRIPTION:INTERNAL=Shape descriptors and matchers.
//List of header files for opencv_shape
OPENCV_MODULE_opencv_shape_HEADERS:INTERNAL=F:/opencv/sources/modules/shape/include/opencv2/shape.hpp;F:/opencv/sources/modules/shape/include/opencv2/shape/emdL1.hpp;F:/opencv/sources/modules/shape/include/opencv2/shape/hist_cost.hpp;F:/opencv/sources/modules/shape/include/opencv2/shape/shape.hpp;F:/opencv/sources/modules/shape/include/opencv2/shape/shape_distance.hpp;F:/opencv/sources/modules/shape/include/opencv2/shape/shape_transformer.hpp
OPENCV_MODULE_opencv_shape_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_shape_LABEL:INTERNAL=Main;opencv_shape
OPENCV_MODULE_opencv_shape_LINK_DEPS:INTERNAL=;
//Location of opencv_shape module sources
OPENCV_MODULE_opencv_shape_LOCATION:INTERNAL=F:/opencv/sources/modules/shape
//Optional dependencies of opencv_shape module
OPENCV_MODULE_opencv_shape_OPT_DEPS:INTERNAL=
//Optional private dependencies of opencv_shape module
OPENCV_MODULE_opencv_shape_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_shape module
OPENCV_MODULE_opencv_shape_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_shape module
OPENCV_MODULE_opencv_shape_REQ_DEPS:INTERNAL=opencv_core;opencv_imgproc;opencv_video
//List of source files for opencv_shape
OPENCV_MODULE_opencv_shape_SOURCES:INTERNAL=F:/opencv/sources/modules/shape/src/aff_trans.cpp;F:/opencv/sources/modules/shape/src/emdL1.cpp;F:/opencv/sources/modules/shape/src/haus_dis.cpp;F:/opencv/sources/modules/shape/src/hist_cost.cpp;F:/opencv/sources/modules/shape/src/precomp.cpp;F:/opencv/sources/modules/shape/src/sc_dis.cpp;F:/opencv/sources/modules/shape/src/tps_trans.cpp;F:/opencv/sources/modules/shape/src/emdL1_def.hpp;F:/opencv/sources/modules/shape/src/precomp.hpp;F:/opencv/sources/modules/shape/src/scd_def.hpp
//List of wrappers supporting module opencv_shape
OPENCV_MODULE_opencv_shape_WRAPPERS:INTERNAL=python2;python3
OPENCV_MODULE_opencv_stitching_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/stitching
//List of opencv_stitching submodules
OPENCV_MODULE_opencv_stitching_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_stitching_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_stitching module
OPENCV_MODULE_opencv_stitching_DEPS:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_imgcodecs;opencv_videoio;opencv_highgui;opencv_objdetect;opencv_features2d;opencv_calib3d
//Extra dependencies of opencv_stitching module
OPENCV_MODULE_opencv_stitching_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_stitching module (for linker)
OPENCV_MODULE_opencv_stitching_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_imgcodecs;opencv_videoio;opencv_highgui;opencv_objdetect;opencv_features2d;opencv_calib3d
//Brief description of opencv_stitching module
OPENCV_MODULE_opencv_stitching_DESCRIPTION:INTERNAL=Images stitching
//List of header files for opencv_stitching
OPENCV_MODULE_opencv_stitching_HEADERS:INTERNAL=F:/opencv/sources/modules/stitching/include/opencv2/stitching.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/warpers.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/autocalib.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/blenders.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/camera.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/exposure_compensate.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/matchers.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/motion_estimators.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/seam_finders.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/timelapsers.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/util.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/util_inl.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/warpers.hpp;F:/opencv/sources/modules/stitching/include/opencv2/stitching/detail/warpers_inl.hpp
OPENCV_MODULE_opencv_stitching_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_stitching_LABEL:INTERNAL=Main;opencv_stitching
OPENCV_MODULE_opencv_stitching_LINK_DEPS:INTERNAL=;
//Location of opencv_stitching module sources
OPENCV_MODULE_opencv_stitching_LOCATION:INTERNAL=F:/opencv/sources/modules/stitching
//Optional dependencies of opencv_stitching module
OPENCV_MODULE_opencv_stitching_OPT_DEPS:INTERNAL=opencv_cudaarithm;opencv_cudafilters;opencv_cudafeatures2d;opencv_cudalegacy;opencv_xfeatures2d
//Optional private dependencies of opencv_stitching module
OPENCV_MODULE_opencv_stitching_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_stitching module
OPENCV_MODULE_opencv_stitching_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_stitching module
OPENCV_MODULE_opencv_stitching_REQ_DEPS:INTERNAL=opencv_imgproc;opencv_features2d;opencv_calib3d;opencv_objdetect
//List of source files for opencv_stitching
OPENCV_MODULE_opencv_stitching_SOURCES:INTERNAL=F:/opencv/sources/modules/stitching/src/autocalib.cpp;F:/opencv/sources/modules/stitching/src/blenders.cpp;F:/opencv/sources/modules/stitching/src/camera.cpp;F:/opencv/sources/modules/stitching/src/exposure_compensate.cpp;F:/opencv/sources/modules/stitching/src/matchers.cpp;F:/opencv/sources/modules/stitching/src/motion_estimators.cpp;F:/opencv/sources/modules/stitching/src/seam_finders.cpp;F:/opencv/sources/modules/stitching/src/stitcher.cpp;F:/opencv/sources/modules/stitching/src/timelapsers.cpp;F:/opencv/sources/modules/stitching/src/util.cpp;F:/opencv/sources/modules/stitching/src/warpers.cpp;F:/opencv/sources/modules/stitching/src/warpers_cuda.cpp;F:/opencv/sources/modules/stitching/src/opencl/multibandblend.cl;F:/opencv/sources/modules/stitching/src/opencl/warpers.cl;F:/opencv/sources/opencv3/modules/stitching/opencl_kernels_stitching.cpp;F:/opencv/sources/opencv3/modules/stitching/opencl_kernels_stitching.hpp;F:/opencv/sources/modules/stitching/src/precomp.hpp;F:/opencv/sources/modules/stitching/src/util_log.hpp
//List of wrappers supporting module opencv_stitching
OPENCV_MODULE_opencv_stitching_WRAPPERS:INTERNAL=python2;python3
OPENCV_MODULE_opencv_superres_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/superres
//List of opencv_superres submodules
OPENCV_MODULE_opencv_superres_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_superres_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_superres module
OPENCV_MODULE_opencv_superres_DEPS:INTERNAL=opencv_core;opencv_imgproc;opencv_video;opencv_imgcodecs;opencv_videoio
//Extra dependencies of opencv_superres module
OPENCV_MODULE_opencv_superres_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_superres module (for linker)
OPENCV_MODULE_opencv_superres_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_imgproc;opencv_video;opencv_imgcodecs;opencv_videoio
//Brief description of opencv_superres module
OPENCV_MODULE_opencv_superres_DESCRIPTION:INTERNAL=Super Resolution
//List of header files for opencv_superres
OPENCV_MODULE_opencv_superres_HEADERS:INTERNAL=F:/opencv/sources/modules/superres/include/opencv2/superres.hpp;F:/opencv/sources/modules/superres/include/opencv2/superres/optical_flow.hpp
OPENCV_MODULE_opencv_superres_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_superres_LABEL:INTERNAL=Main;opencv_superres
OPENCV_MODULE_opencv_superres_LINK_DEPS:INTERNAL=;
//Location of opencv_superres module sources
OPENCV_MODULE_opencv_superres_LOCATION:INTERNAL=F:/opencv/sources/modules/superres
//Optional dependencies of opencv_superres module
OPENCV_MODULE_opencv_superres_OPT_DEPS:INTERNAL=opencv_videoio;opencv_cudaarithm;opencv_cudafilters;opencv_cudawarping;opencv_cudaimgproc;opencv_cudaoptflow;opencv_cudacodec
//Optional private dependencies of opencv_superres module
OPENCV_MODULE_opencv_superres_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_superres module
OPENCV_MODULE_opencv_superres_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_superres module
OPENCV_MODULE_opencv_superres_REQ_DEPS:INTERNAL=opencv_imgproc;opencv_video
//List of source files for opencv_superres
OPENCV_MODULE_opencv_superres_SOURCES:INTERNAL=F:/opencv/sources/modules/superres/src/btv_l1.cpp;F:/opencv/sources/modules/superres/src/btv_l1_cuda.cpp;F:/opencv/sources/modules/superres/src/frame_source.cpp;F:/opencv/sources/modules/superres/src/input_array_utility.cpp;F:/opencv/sources/modules/superres/src/optical_flow.cpp;F:/opencv/sources/modules/superres/src/super_resolution.cpp;F:/opencv/sources/modules/superres/src/opencl/superres_btvl1.cl;F:/opencv/sources/opencv3/modules/superres/opencl_kernels_superres.cpp;F:/opencv/sources/opencv3/modules/superres/opencl_kernels_superres.hpp;F:/opencv/sources/modules/superres/src/input_array_utility.hpp;F:/opencv/sources/modules/superres/src/precomp.hpp;F:/opencv/sources/modules/superres/src/ring_buffer.hpp
//List of wrappers supporting module opencv_superres
OPENCV_MODULE_opencv_superres_WRAPPERS:INTERNAL=python2;python3
OPENCV_MODULE_opencv_ts_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/ts
//List of opencv_ts submodules
OPENCV_MODULE_opencv_ts_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_ts_CLASS:INTERNAL=INTERNAL
//Flattened dependencies of opencv_ts module
OPENCV_MODULE_opencv_ts_DEPS:INTERNAL=opencv_core;opencv_imgproc;opencv_imgcodecs;opencv_videoio;opencv_highgui
//Extra dependencies of opencv_ts module
OPENCV_MODULE_opencv_ts_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_ts module (for linker)
OPENCV_MODULE_opencv_ts_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_imgproc;opencv_imgcodecs;opencv_videoio;opencv_highgui
//Brief description of opencv_ts module
OPENCV_MODULE_opencv_ts_DESCRIPTION:INTERNAL=The ts module
//List of header files for opencv_ts
OPENCV_MODULE_opencv_ts_HEADERS:INTERNAL=F:/opencv/sources/modules/ts/include/opencv2/ts.hpp;F:/opencv/sources/modules/ts/include/opencv2/ts/cuda_perf.hpp;F:/opencv/sources/modules/ts/include/opencv2/ts/cuda_test.hpp;F:/opencv/sources/modules/ts/include/opencv2/ts/ocl_perf.hpp;F:/opencv/sources/modules/ts/include/opencv2/ts/ocl_test.hpp;F:/opencv/sources/modules/ts/include/opencv2/ts/ts_ext.hpp;F:/opencv/sources/modules/ts/include/opencv2/ts/ts_perf.hpp;F:/opencv/sources/modules/ts/include/opencv2/ts/ts_gtest.h
OPENCV_MODULE_opencv_ts_IS_PART_OF_WORLD:INTERNAL=OFF
OPENCV_MODULE_opencv_ts_LABEL:INTERNAL=Main;opencv_ts
OPENCV_MODULE_opencv_ts_LINK_DEPS:INTERNAL=;
//Location of opencv_ts module sources
OPENCV_MODULE_opencv_ts_LOCATION:INTERNAL=F:/opencv/sources/modules/ts
//Optional dependencies of opencv_ts module
OPENCV_MODULE_opencv_ts_OPT_DEPS:INTERNAL=
//Optional private dependencies of opencv_ts module
OPENCV_MODULE_opencv_ts_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_ts module
OPENCV_MODULE_opencv_ts_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_ts module
OPENCV_MODULE_opencv_ts_REQ_DEPS:INTERNAL=opencv_core;opencv_imgproc;opencv_imgcodecs;opencv_videoio;opencv_highgui
//List of source files for opencv_ts
OPENCV_MODULE_opencv_ts_SOURCES:INTERNAL=F:/opencv/sources/modules/ts/src/cuda_perf.cpp;F:/opencv/sources/modules/ts/src/cuda_test.cpp;F:/opencv/sources/modules/ts/src/ocl_perf.cpp;F:/opencv/sources/modules/ts/src/ocl_test.cpp;F:/opencv/sources/modules/ts/src/ts.cpp;F:/opencv/sources/modules/ts/src/ts_arrtest.cpp;F:/opencv/sources/modules/ts/src/ts_func.cpp;F:/opencv/sources/modules/ts/src/ts_gtest.cpp;F:/opencv/sources/modules/ts/src/ts_perf.cpp;F:/opencv/sources/modules/ts/src/precomp.hpp
//List of wrappers supporting module opencv_ts
OPENCV_MODULE_opencv_ts_WRAPPERS:INTERNAL=
OPENCV_MODULE_opencv_video_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/video
//List of opencv_video submodules
OPENCV_MODULE_opencv_video_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_video_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_video module
OPENCV_MODULE_opencv_video_DEPS:INTERNAL=opencv_core;opencv_imgproc
//Extra dependencies of opencv_video module
OPENCV_MODULE_opencv_video_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_video module (for linker)
OPENCV_MODULE_opencv_video_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_imgproc
//Brief description of opencv_video module
OPENCV_MODULE_opencv_video_DESCRIPTION:INTERNAL=Video Analysis
//List of header files for opencv_video
OPENCV_MODULE_opencv_video_HEADERS:INTERNAL=F:/opencv/sources/modules/video/include/opencv2/video.hpp;F:/opencv/sources/modules/video/include/opencv2/video/background_segm.hpp;F:/opencv/sources/modules/video/include/opencv2/video/tracking.hpp;F:/opencv/sources/modules/video/include/opencv2/video/video.hpp;F:/opencv/sources/modules/video/include/opencv2/video/tracking_c.h
OPENCV_MODULE_opencv_video_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_video_LABEL:INTERNAL=Main;opencv_video
OPENCV_MODULE_opencv_video_LINK_DEPS:INTERNAL=;
//Location of opencv_video module sources
OPENCV_MODULE_opencv_video_LOCATION:INTERNAL=F:/opencv/sources/modules/video
//Optional dependencies of opencv_video module
OPENCV_MODULE_opencv_video_OPT_DEPS:INTERNAL=
//Optional private dependencies of opencv_video module
OPENCV_MODULE_opencv_video_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_video module
OPENCV_MODULE_opencv_video_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_video module
OPENCV_MODULE_opencv_video_REQ_DEPS:INTERNAL=opencv_imgproc
//List of source files for opencv_video
OPENCV_MODULE_opencv_video_SOURCES:INTERNAL=F:/opencv/sources/modules/video/src/bgfg_KNN.cpp;F:/opencv/sources/modules/video/src/bgfg_gaussmix2.cpp;F:/opencv/sources/modules/video/src/camshift.cpp;F:/opencv/sources/modules/video/src/compat_video.cpp;F:/opencv/sources/modules/video/src/ecc.cpp;F:/opencv/sources/modules/video/src/kalman.cpp;F:/opencv/sources/modules/video/src/lkpyramid.cpp;F:/opencv/sources/modules/video/src/optflowgf.cpp;F:/opencv/sources/modules/video/src/tvl1flow.cpp;F:/opencv/sources/modules/video/src/opencl/bgfg_mog2.cl;F:/opencv/sources/modules/video/src/opencl/optical_flow_farneback.cl;F:/opencv/sources/modules/video/src/opencl/optical_flow_tvl1.cl;F:/opencv/sources/modules/video/src/opencl/pyrlk.cl;F:/opencv/sources/opencv3/modules/video/opencl_kernels_video.cpp;F:/opencv/sources/opencv3/modules/video/opencl_kernels_video.hpp;F:/opencv/sources/modules/video/src/lkpyramid.hpp;F:/opencv/sources/modules/video/src/precomp.hpp
//List of wrappers supporting module opencv_video
OPENCV_MODULE_opencv_video_WRAPPERS:INTERNAL=java;python2;python3
OPENCV_MODULE_opencv_videoio_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/videoio
//List of opencv_videoio submodules
OPENCV_MODULE_opencv_videoio_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_videoio_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_videoio module
OPENCV_MODULE_opencv_videoio_DEPS:INTERNAL=opencv_core;opencv_imgproc;opencv_imgcodecs
//Extra dependencies of opencv_videoio module
OPENCV_MODULE_opencv_videoio_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_videoio module (for linker)
OPENCV_MODULE_opencv_videoio_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_imgproc;opencv_imgcodecs
//Brief description of opencv_videoio module
OPENCV_MODULE_opencv_videoio_DESCRIPTION:INTERNAL=Media I/O
//List of header files for opencv_videoio
OPENCV_MODULE_opencv_videoio_HEADERS:INTERNAL=F:/opencv/sources/modules/videoio/include/opencv2/videoio.hpp;F:/opencv/sources/modules/videoio/include/opencv2/videoio/videoio.hpp;F:/opencv/sources/modules/videoio/include/opencv2/videoio/cap_ios.h;F:/opencv/sources/modules/videoio/include/opencv2/videoio/videoio_c.h
OPENCV_MODULE_opencv_videoio_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_videoio_LABEL:INTERNAL=Main;opencv_videoio
OPENCV_MODULE_opencv_videoio_LINK_DEPS:INTERNAL=;vfw32;winmm
//Location of opencv_videoio module sources
OPENCV_MODULE_opencv_videoio_LOCATION:INTERNAL=F:/opencv/sources/modules/videoio
//Optional dependencies of opencv_videoio module
OPENCV_MODULE_opencv_videoio_OPT_DEPS:INTERNAL=
//Optional private dependencies of opencv_videoio module
OPENCV_MODULE_opencv_videoio_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_videoio module
OPENCV_MODULE_opencv_videoio_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_videoio module
OPENCV_MODULE_opencv_videoio_REQ_DEPS:INTERNAL=opencv_imgproc;opencv_imgcodecs
//List of source files for opencv_videoio
OPENCV_MODULE_opencv_videoio_SOURCES:INTERNAL=F:/opencv/sources/modules/videoio/src/cap.cpp;F:/opencv/sources/modules/videoio/src/cap_images.cpp;F:/opencv/sources/modules/videoio/src/cap_mjpeg_encoder.cpp;F:/opencv/sources/modules/videoio/src/cap_mjpeg_decoder.cpp;F:/opencv/sources/modules/videoio/src/cap_cmu.cpp;F:/opencv/sources/modules/videoio/src/cap_dshow.cpp;F:/opencv/sources/modules/videoio/src/cap_vfw.cpp;F:/opencv/sources/modules/videoio/src/cap_ffmpeg.cpp;F:/opencv/sources/modules/videoio/src/precomp.hpp;F:/opencv/sources/modules/videoio/src/cap_dshow.hpp;F:/opencv/sources/modules/videoio/src/cap_ffmpeg_impl.hpp
//List of wrappers supporting module opencv_videoio
OPENCV_MODULE_opencv_videoio_WRAPPERS:INTERNAL=java;python2;python3
OPENCV_MODULE_opencv_videostab_BINARY_DIR:INTERNAL=F:/opencv/sources/opencv3/modules/videostab
//List of opencv_videostab submodules
OPENCV_MODULE_opencv_videostab_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_videostab_CLASS:INTERNAL=PUBLIC
//Flattened dependencies of opencv_videostab module
OPENCV_MODULE_opencv_videostab_DEPS:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_photo;opencv_video;opencv_imgcodecs;opencv_videoio;opencv_highgui;opencv_features2d;opencv_calib3d
//Extra dependencies of opencv_videostab module
OPENCV_MODULE_opencv_videostab_DEPS_EXT:INTERNAL=
//Flattened dependencies of opencv_videostab module (for linker)
OPENCV_MODULE_opencv_videostab_DEPS_TO_LINK:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_photo;opencv_video;opencv_imgcodecs;opencv_videoio;opencv_highgui;opencv_features2d;opencv_calib3d
//Brief description of opencv_videostab module
OPENCV_MODULE_opencv_videostab_DESCRIPTION:INTERNAL=Video stabilization
//List of header files for opencv_videostab
OPENCV_MODULE_opencv_videostab_HEADERS:INTERNAL=F:/opencv/sources/modules/videostab/include/opencv2/videostab.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/deblurring.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/fast_marching.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/fast_marching_inl.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/frame_source.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/global_motion.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/inpainting.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/log.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/motion_core.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/motion_stabilizing.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/optical_flow.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/outlier_rejection.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/ring_buffer.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/stabilizer.hpp;F:/opencv/sources/modules/videostab/include/opencv2/videostab/wobble_suppression.hpp
OPENCV_MODULE_opencv_videostab_IS_PART_OF_WORLD:INTERNAL=ON
OPENCV_MODULE_opencv_videostab_LABEL:INTERNAL=Main;opencv_videostab
OPENCV_MODULE_opencv_videostab_LINK_DEPS:INTERNAL=;
//Location of opencv_videostab module sources
OPENCV_MODULE_opencv_videostab_LOCATION:INTERNAL=F:/opencv/sources/modules/videostab
//Optional dependencies of opencv_videostab module
OPENCV_MODULE_opencv_videostab_OPT_DEPS:INTERNAL=opencv_cudawarping;opencv_cudaoptflow;opencv_videoio
//Optional private dependencies of opencv_videostab module
OPENCV_MODULE_opencv_videostab_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_videostab module
OPENCV_MODULE_opencv_videostab_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_videostab module
OPENCV_MODULE_opencv_videostab_REQ_DEPS:INTERNAL=opencv_imgproc;opencv_features2d;opencv_video;opencv_photo;opencv_calib3d
//List of source files for opencv_videostab
OPENCV_MODULE_opencv_videostab_SOURCES:INTERNAL=F:/opencv/sources/modules/videostab/src/deblurring.cpp;F:/opencv/sources/modules/videostab/src/fast_marching.cpp;F:/opencv/sources/modules/videostab/src/frame_source.cpp;F:/opencv/sources/modules/videostab/src/global_motion.cpp;F:/opencv/sources/modules/videostab/src/inpainting.cpp;F:/opencv/sources/modules/videostab/src/log.cpp;F:/opencv/sources/modules/videostab/src/motion_stabilizing.cpp;F:/opencv/sources/modules/videostab/src/optical_flow.cpp;F:/opencv/sources/modules/videostab/src/outlier_rejection.cpp;F:/opencv/sources/modules/videostab/src/stabilizer.cpp;F:/opencv/sources/modules/videostab/src/wobble_suppression.cpp;F:/opencv/sources/modules/videostab/src/clp.hpp;F:/opencv/sources/modules/videostab/src/precomp.hpp
//List of wrappers supporting module opencv_videostab
OPENCV_MODULE_opencv_videostab_WRAPPERS:INTERNAL=python2;python3
//Location of opencv_viz module sources
OPENCV_MODULE_opencv_viz_LOCATION:INTERNAL=F:/opencv/sources/modules/viz
//List of opencv_world submodules
OPENCV_MODULE_opencv_world_CHILDREN:INTERNAL=
//The category of the module
OPENCV_MODULE_opencv_world_CLASS:INTERNAL=PUBLIC
//Brief description of opencv_world module
OPENCV_MODULE_opencv_world_DESCRIPTION:INTERNAL=All the selected OpenCV modules in a single binary
OPENCV_MODULE_opencv_world_IS_PART_OF_WORLD:INTERNAL=OFF
OPENCV_MODULE_opencv_world_LABEL:INTERNAL=Main;opencv_world
OPENCV_MODULE_opencv_world_LINK_DEPS:INTERNAL=
//Location of opencv_world module sources
OPENCV_MODULE_opencv_world_LOCATION:INTERNAL=F:/opencv/sources/modules/world
//Optional dependencies of opencv_world module
OPENCV_MODULE_opencv_world_OPT_DEPS:INTERNAL=opencv_calib3d;opencv_core;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab
//Optional private dependencies of opencv_world module
OPENCV_MODULE_opencv_world_PRIVATE_OPT_DEPS:INTERNAL=
//Required private dependencies of opencv_world module
OPENCV_MODULE_opencv_world_PRIVATE_REQ_DEPS:INTERNAL=
//Required dependencies of opencv_world module
OPENCV_MODULE_opencv_world_REQ_DEPS:INTERNAL=opencv_core
//List of wrappers supporting module opencv_world
OPENCV_MODULE_opencv_world_WRAPPERS:INTERNAL=
//List of OpenCV modules included into the world
OPENCV_WORLD_MODULES:INTERNAL=
//ADVANCED property for variable: OpenBLAS_INCLUDE_DIR
OpenBLAS_INCLUDE_DIR-ADVANCED:INTERNAL=1
//ADVANCED property for variable: OpenBLAS_LIB
OpenBLAS_LIB-ADVANCED:INTERNAL=1
//List of OpenCVModules targets
OpenCVModules_TARGETS:INTERNAL=opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_photo;opencv_video;opencv_imgcodecs;opencv_shape;opencv_videoio;opencv_highgui;opencv_objdetect;opencv_superres;opencv_features2d;opencv_calib3d;opencv_java;opencv_stitching;opencv_videostab
PYTHON2INTERP_FOUND:INTERNAL=TRUE
PYTHON2LIBS_FOUND:INTERNAL=TRUE
PYTHON2LIBS_VERSION_STRING:INTERNAL=2.7.10+
PYTHON2_DEBUG_LIBRARIES:INTERNAL=PYTHON_DEBUG_LIBRARY-NOTFOUND
PYTHON2_INCLUDE_PATH:INTERNAL=C:/yotta/python/include
//Python libraries
PYTHON2_LIBRARIES:INTERNAL=C:/yotta/python/libs/libpython27.a
PYTHON2_NUMPY_VERSION:INTERNAL=
PYTHON2_VERSION_MAJOR:INTERNAL=2
PYTHON2_VERSION_MINOR:INTERNAL=7
PYTHON2_VERSION_STRING:INTERNAL=2.7.10
PYTHON3INTERP_FOUND:INTERNAL=
PYTHON3LIBS_FOUND:INTERNAL=
PYTHON3LIBS_VERSION_STRING:INTERNAL=
PYTHON3_DEBUG_LIBRARIES:INTERNAL=
PYTHON3_INCLUDE_PATH:INTERNAL=
//Python libraries
PYTHON3_LIBRARIES:INTERNAL=
PYTHON3_NUMPY_VERSION:INTERNAL=
PYTHON3_VERSION_MAJOR:INTERNAL=
PYTHON3_VERSION_MINOR:INTERNAL=
PYTHON3_VERSION_STRING:INTERNAL=
//ADVANCED property for variable: PYTHON_EXECUTABLE
PYTHON_EXECUTABLE-ADVANCED:INTERNAL=1
//Result of TEST_BIG_ENDIAN
WORDS_BIGENDIAN:INTERNAL=0
//Have include unistd.h
Z_HAVE_UNISTD_H:INTERNAL=1
//Result of TRY_COMPILE
__VALID_DIRECTX:INTERNAL=FALSE
//Result of TRY_COMPILE
__VALID_FP16:INTERNAL=FALSE
//Result of TRY_COMPILE
__VALID_OPENCL:INTERNAL=FALSE
```






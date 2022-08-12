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

## 3.Jetson AGX Xavier安装opencv3.4支持cuda和gstreamer

系统是 Jetpack 4.5.1，默认安装 opencv4.4.1，为了重新安装支持cuda的opencv，把当前的opencv卸载了,sudo apt purge libopencv*,然后再查看jetson_release：

![image-20220714101458831](images/image-20220714101458831.png)

重新安装：

参考链接1：https://blog.csdn.net/m0_62013374/article/details/125014338

参考链接2：http://t.zoukankan.com/gloria-zhang-p-13819297.html  (推荐)

参考链接3:https://blog.csdn.net/kittyzc/article/details/117388718（用来测试gstreamer是否正常）

参考链接4::https://blog.csdn.net/quicmous/article/details/124172950

- 下载源码

  opencv的下载地址：https://github.com/opencv/opencv/

  opencv_contrib的下载地址：https://github.com/opencv/opencv_contrib.git

  对opencv_contrib的解释：You can build OpenCV, so it will include the modules from this repository. Contrib modules are under constant development and it is recommended to use them alongside the master branch or latest releases of OpenCV.

  opencv和opencv_contrib选择相同得版本(即git相同或相近号码的分支)，复制code的zip包链接，

  在机器上wget下来解压。

  这里是将contrib文件夹移到了opencv里面(放哪都行，编译的时候指定文件夹位置即可)

- 安装依赖

  ```
  sudo apt-get update
  sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
  #gstreamer
  sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  #gstreamer1.0-plugins-bad ，这个包含了 h264parse 元素. 这是为了解码来自 IP 摄像头的 H.264 RTSP stream 所需要的，本项目使用的流编码格式为h264
  sudo apt-get install gstreamer1.0-plugins-bad
  #python
  sudo apt-get install -y python2.7-dev
  sudo apt-get install -y python3-dev python3-numpy python3-py python3-pytest
  sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
  sudo apt-get install -y libv4l-dev v4l-utils qv4l2 v4l2ucp
  sudo apt-get install -y curl
  sudo apt-get update
  ```

- 编译

  在opencv源码文件夹内进行如下操作：

  ```
  mkdir release
  cd release/
  #cmake编译，产生build文件,注意-DWITH_GSTREAMER = ON，确保OpenCV编译时使用gstreamer支持
  #默认是基于Python2.7，使用Python3需要加上-DBUILD_opencv_python3=ON
  cmake -D WITH_CUDA=ON -D CUDA_ARCH_BIN="7.2"  -D CUDA_ARCH_PTX="" -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local OPENCV_EXTRA_MODULES_PATH=./opencv_contrib-3.4.3/modules ..
  #make利用CPU多核编译
  make -j8
  #最后安装
  sudo make install
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

Pad是一个element的输入/输出接口，分为src pad（生产数据）和sink pad（消费数据）两种。
两个element必须通过pad才能连接起来，pad拥有当前element能处理数据类型的能力（capabilities），会在连接时通过比较src pad和sink pad中所支持的能力，来选择最恰当的数据类型用于传输，如果element不支持，程序会直接退出。在element通过pad连接成功后，数据会从上一个element的src pad传到下一个element的sink pad然后进行处理。
当element支持多种数据处理能力时，我们可以通过Cap来指定数据类型.
例如，下面的命令通过Cap指定了视频的宽高，videotestsrc会根据指定的宽高产生相应数据：

gst-launch-1.0 videotestsrc ! "video/x-raw,width=1280,height=720" ! autovideosink

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
通过上面ogg视频播放的例子，我们已经看到，一个pipeline的多个element之间通过 “!" 分隔，同时可以设置element及Cap的属性。例如：
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




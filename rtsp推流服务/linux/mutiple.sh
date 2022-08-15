#!/usr/bin/bash
for i in {0..100}
do
  {
    nohup /disks/sdc/eg1/ffmpeg-git-20220302-amd64-static/ffmpeg -re -stream_loop -1 -i /disks/sdc/eg1/videos/tongdao.mp4 -c copy -f rtsp rtsp://127.0.0.1:18554/stream$i >/dev/null 2>&1 &
  }
done
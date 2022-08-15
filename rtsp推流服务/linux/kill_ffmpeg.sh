#!/usr/bin/bash
ps -ef|grep ffmpeg|grep -v grep|cut -c 9-16|xargs kill -9
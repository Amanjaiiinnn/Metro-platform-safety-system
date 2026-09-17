#!/bin/sh
export DISPLAY=:0
export XDG_RUNTIME_DIR=/run/user/0

cd /root/metro || exit 1

sleep 10

exec /usr/bin/python3 -u /root/metro/launcher.py

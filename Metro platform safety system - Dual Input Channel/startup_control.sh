#!/bin/sh
export DISPLAY=:0
export XDG_RUNTIME_DIR=/run/user/0

if [ -d "/root/metro2" ]; then
    cd /root/metro2 || exit 1
elif [ -d "/root/metro" ]; then
    cd /root/metro || exit 1
fi

sleep 5

exec /usr/bin/python3 -u launcher.py

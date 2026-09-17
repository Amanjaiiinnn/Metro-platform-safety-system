#!/usr/bin/env python3

"""
Copyright 2026 EDS

This script launches the EDS-DMS Demo 
"""

import os
import subprocess
import threading

from config import (
    YOLO_MODEL,
    CLOTHING_MODEL,
    POSE_MODEL,
    YOLO_VELA_MODEL,
    CLOTHING_VELA_MODEL,
    POSE_VELA_MODEL,
    BACKEND,
    YOLO_NEUTRON_MODEL,
    POSE_NEUTRON_MODEL,
    CLOTHING_NEUTRON_MODEL,
)

cur_path = os.path.dirname(os.path.abspath(__file__))
print(cur_path)

MODEL_DIR = "/root/metro/models/"

def threaded(fn):
    """
    Handle threads out of main GTK thread
    """

    def wrapper(*args, ** kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs).start()

    return wrapper

class MetroSafetyLauncher:
    """
    EDS-Metro Safety launcher
    """

    def __init__(self):
        self.platform = None

        # Check Target (i.MX 93 vs i.MX 95 vs PC)
        if os.path.exists("/usr/lib/libethosu_delegate.so"):
            self.platform = "i.MX93"
        elif os.path.exists("/usr/lib/libneutron_delegate.so"):
            self.platform = "i.MX95"
        else:
            self.platform = "PC"
        
        print(f"Target is {self.platform}")
        
        # Define names and models
        self.yolo_model = YOLO_MODEL
        self.clothing_model = CLOTHING_MODEL
        self.pose_model = POSE_MODEL

    @threaded
    def start(self, widget):
        """
        Funtion to start and run Code
        """
        if(self.platform == "PC"):
            backend = "CPU"
        else:
            backend = BACKEND
        print("Loading the Models to Cache...")

        # Load Models and save graph on cache
        if(self.platform == "i.MX93" and backend == "NPU"):
            # overwrite models name if backend is NPU for imx93
            yolo_vela_model = YOLO_VELA_MODEL
            clothing_vela_model = CLOTHING_VELA_MODEL
            pose_vela_model = POSE_VELA_MODEL

            # print(MODEL_DIR + yolo_vela_model)
            # print(MODEL_DIR + clothing_vela_model)

            if not os.path.exists(MODEL_DIR + yolo_vela_model):
                print("Compling and saving Yolo model to cache...")
            
                subprocess.run(
                    "vela /root/metro/models/"
                    + self.yolo_model 
                    + " --output-dir=/root/metro/models/",
                    shell=True,
                    check=True,
                )
            
            if not os.path.exists(MODEL_DIR + clothing_vela_model):
                print("Compling and saving Clothing model to cache...")
            
                subprocess.run(
                    "vela /root/metro/models/"
                    + self.clothing_model
                    + " --output-dir=/root/metro/models/",
                    shell=True,
                    check=True,
                )

            if not os.path.exists(MODEL_DIR + pose_vela_model):
                print("Compling and saving Pose model to cache...")

                subprocess.run(
                    "vela /root/metro/models/"
                    + self.pose_model
                    + " --output-dir=/root/metro/models/",
                    shell=True,
                    check=True,
                )
        
        if(self.platform == "i.MX95" and backend == "NPU"):
            # overwrite models name if backend is NPU for imx93
            yolo_neutron_model = YOLO_NEUTRON_MODEL
            clothing_neutron_model = CLOTHING_NEUTRON_MODEL
            pose_neutron_model = POSE_NEUTRON_MODEL

            if not os.path.exists(MODEL_DIR + yolo_neutron_model):
                print(f"{MODEL_DIR + yolo_neutron_model} not available in the file.")
            
            if not os.path.exists(MODEL_DIR + clothing_neutron_model):
                print(f"{MODEL_DIR + clothing_neutron_model} not available in the file.")

            if not os.path.exists(MODEL_DIR + pose_neutron_model):
                print(f"{MODEL_DIR + pose_neutron_model} not available in the file.")
        
        print("Models are Ready!")

        subprocess.run(
            "python3 "
            + cur_path
            + "/main.py"
            + f" --platform \"{self.platform}\" --backend \"{backend}\"",
            shell=True,
            check=True,
        )

        return True


if __name__ == "__main__":
    launcher = MetroSafetyLauncher()
    launcher.start(None)
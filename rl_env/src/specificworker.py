#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2022 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import time
from zmqRemoteApi import RemoteAPIClient

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)

        self.constants = {
            "EPOCHS": 3,
            "SCENE": "/home/robocomp/robocomp/components/manipulation_kinova_gen3/etc/kinova_env_dani_bloquesApilados.ttt",
            "TIMESTAMPS_PER_EPOCH": 2,
            "POS": 1,
            "ZERO": 0,
            "NEG": -1,
            "ARM_PATH": "/customizableTable/gen3",
            "CAMERA_PATH": "/customizableTable/Actuator8/\
                            Shoulder_Link_respondable0/Actuator0/\
                            HalfArm1_Link_respondable0/Actuator2/\
                            HalfArm2_Link_respondable/Actuator3/\
                            ForeArm_Link_respondable0/Actuator14/\
                            SphericalWrist1_link_respondable0/Actuator5/\
                            SphericalWrist2_Link_respondable0/Actuator6/\
                            Bracelet_Link_respondable0/Bracelet_link_visual0/camera_arm"
        }
        
        print("Program started")
        client = RemoteAPIClient()
        self.sim = client.getObject("sim")
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.loadScene(self.constants["SCENE"])
        print('Scene loaded')

        self.arm = self.sim.getObject(self.constants["ARM_PATH"])
        self.camera = self.sim.getObject(self.constants["CAMERA_PATH"])

        #client.setStepping(True)
        self.sim.startSimulation()

        self.Period = 2000
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        print("Program ended")

    def setParams(self, params):
        return True


    @QtCore.Slot()
    def compute(self):
        print("SpecificWorker.compute...")

        return True

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    def step(action):
        # COPPELIA STEP
        pass

    def reset():
        pass

    def next_action():
        pass

    def reward():
        pass 


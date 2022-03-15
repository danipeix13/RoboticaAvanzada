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

from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QTimer
from rich.console import Console
import interfaces as ifaces
from genericworker import *
# from EnvKinova import *

from EnvKinova_gym import *
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        print("SpecificWorker.__init__")

        self.env = EnvKinova_gym()
        self.model = PPO("MlpPolicy", self.env, learning_rate=1e-2, verbose=1)
        time.sleep(1)
        check_env(self.env, warn=True)
        self.model.learn(total_timesteps=30000)
        
        self.obs = self.env.reset()
        self.Period = 250
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        print("SpecificWorker.__del__")
        self.env.close()
        pass

    def setParams(self, params):
        print("SpecificWorker.setParams")
        return True

    @QtCore.Slot()
    def compute(self):
        print("\nSpecificWorker.compute...")

        action, _ = self.model.predict(self.obs)
        self.obs, reward, done, info = self.env.step(action)
        print('obs=', self.obs, 'reward=', reward, 'done=', done)
        action = self.env.action_space_sample()

        if done:
            self.env.reset()
        self.env.test()
        return True

    def startup_check(self):
        print("SpecificWorker.startup_check")
        QTimer.singleShot(200, QApplication.instance().quit)


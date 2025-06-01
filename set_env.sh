#!/usr/bin/bash
apt-get update
apt-get install libegl-dev libgl1-mesa-glx libosmesa6
export MUJOCO_GL=egl

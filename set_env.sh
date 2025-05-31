#!/usr/bin/bash
sudo apt-get update
sudo apt-get install -y \
    libegl-dev \               # Provides libEGL for EGL contexts
    libosmesa6-dev \           # Off-screen Mesa rendering (OSMesa)
    libgl1-mesa-glx \          # Core Mesa OpenGL (GLX) for fallback
    libglew2.1 \               # OpenGL Extension Wrangler (GLEW) symbols
    libglfw3 \                 # GLFW window/context library
    patchelf                   # Utility for fixing library paths (if needed)
#!/bin/bash

if [[ $# -eq 2 ]]; then
  zig translate-c $1 -D VK_USE_PLATFORM_WAYLAND_KHR -I /usr/include -I /home/nyr/libs/vulkan/1.4.321.1/x86_64/include > $2
else
  echo "To few arguments provided :("
fi

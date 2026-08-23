-- Fluid Motion: one named pipe per mpv instance (load last).
local utils = require "mp.utils"
local pid = utils.getpid()
mp.set_property("input-ipc-server", "\\\\.\\pipe\\fluid-mpv-" .. pid)

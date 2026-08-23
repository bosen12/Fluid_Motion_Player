-- Fluid Motion: named pipe. F3 only notifies the tray app — never injects vf.
local utils = require "mp.utils"
local pid = utils.getpid()
mp.set_property("options/input-ipc-server", "fluid-mpv-" .. pid)

local appdata = os.getenv("APPDATA") or ""
local alive_path = appdata .. "\\FluidMotion\\alive"
local hotkey_path = appdata .. "\\FluidMotion\\hotkey"

local function fluid_on()
  local vf = mp.get_property("vf") or ""
  return vf:find("@fluid", 1, true) or vf:find("fluid_rife", 1, true)
end

local function fluid_alive()
  local f = io.open(alive_path, "r")
  if not f then
    return false
  end
  local raw = f:read("*a")
  f:close()
  local ts = tonumber(raw)
  if not ts then
    return false
  end
  return (os.time() - ts) < 4
end

local function strip_stale()
  if fluid_alive() then
    return
  end
  local vf = mp.get_property("vf") or ""
  if vf:find("@fluid", 1, true) then
    mp.commandv("vf", "remove", "@fluid")
  end
  -- unlabeled leftover from older inject / watch-later
  if vf:find("fluid_rife", 1, true) then
    pcall(function()
      mp.commandv("vf", "remove", "vapoursynth")
    end)
  end
end

local function write_hotkey(action)
  local f = io.open(hotkey_path, "w")
  if not f then
    return false
  end
  f:write(action)
  f:close()
  return true
end

local function toggle_fluid()
  if not fluid_alive() then
    mp.osd_message("請先開啟 Fluid Motion", 2)
    return
  end
  if fluid_on() then
    pcall(function()
      mp.commandv("vf", "remove", "@fluid")
    end)
    write_hotkey("off")
    mp.osd_message("Fluid Motion  關", 1.5)
    return
  end
  if not write_hotkey("on") then
    mp.osd_message("無法通知 Fluid Motion", 2)
    return
  end
  mp.osd_message("Fluid Motion  開", 1.5)
end

mp.add_timeout(0, strip_stale)
mp.register_event("start-file", strip_stale)
mp.register_event("file-loaded", strip_stale)
mp.add_key_binding("F3", "fluid-toggle", toggle_fluid)

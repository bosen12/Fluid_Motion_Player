-- Fluid Motion: named pipe. F3 only notifies the tray app — never injects vf.
local utils = require "mp.utils"
local pid = utils.getpid()
mp.set_property("options/input-ipc-server", "fluid-mpv-" .. pid)

local appdata = os.getenv("APPDATA") or ""
local alive_path = appdata .. "\\FluidMotion\\alive"
local hotkey_path = appdata .. "\\FluidMotion\\hotkey"
local seek_hold_path = appdata .. "\\FluidMotion\\seek_hold"
-- Quiet period after a seek settles before re-applying the filter. Short on
-- purpose: this is just debounce against still-dragging, not real work --
-- the actual rebuild cost is entirely in apply() afterward.
local SEEK_RESUME = 0.15
local seek_timer
local seek_held = false

-- A lone seek is not worth tearing the filter down for. mpv reinitialises the
-- whole VapourSynth script on any seek whether or not the filter is ours to
-- remove, so dropping it buys nothing (measured: 0.69s to remove, seek and
-- re-add versus 0.71s to leave it loaded) while adding SEEK_RESUME plus the
-- tray app's poll interval on top -- 0.2-0.45s of stall that only exists
-- because we reached for the filter.
--
-- Dragging the timeline is the case that does need it gone: every scrub
-- position would otherwise pay a full VS reinit and the drag stops tracking
-- the cursor. So the teardown stays, gated on a second seek landing close
-- behind the first, which is what a drag looks like and a keypress does not.
local SCRUB_WINDOW = 0.5
local last_seek_at = -1e9

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

local function clear_seek_hold()
  os.remove(seek_hold_path)
end

local function touch_seek_hold()
  local f = io.open(seek_hold_path, "w")
  if not f then
    return
  end
  f:write("1")
  f:close()
end

local function begin_seek_hold()
  if not fluid_alive() then
    return false
  end
  if not fluid_on() and not seek_held then
    return false
  end
  if fluid_on() then
    pcall(function()
      mp.commandv("vf", "remove", "@fluid")
    end)
  end
  touch_seek_hold()
  seek_held = true
  if seek_timer then
    seek_timer:kill()
    seek_timer = nil
  end
  return true
end

local function arm_resume()
  if not seek_held then
    return
  end
  if seek_timer then
    seek_timer:kill()
  end
  seek_timer = mp.add_timeout(SEEK_RESUME, function()
    seek_timer = nil
    seek_held = false
    clear_seek_hold()
  end)
end

-- Drop the filter only once this looks like a drag: a second seek arriving
-- within SCRUB_WINDOW of the last one, or a hold that is already open and
-- has not been resumed yet. A single seek falls through and leaves the
-- filter alone, so mpv reinitialises it once and nothing waits on us.
local function on_seek()
  local now = mp.get_time()
  local dragging = (now - last_seek_at) < SCRUB_WINDOW
  last_seek_at = now
  if dragging or seek_held then
    begin_seek_hold()
  end
end

mp.add_timeout(0, strip_stale)
mp.register_event("start-file", strip_stale)
mp.register_event("file-loaded", strip_stale)
-- Drag detection reads the `seek` event only. The `seeking` property flips
-- true for the same seek microseconds later, and letting both feed on_seek()
-- would make every single seek look like a drag against itself. The property
-- observer's job here is just to keep an already-open hold from expiring
-- while the drag is still going.
mp.register_event("seek", on_seek)
mp.register_event("playback-restart", function()
  arm_resume()
end)
mp.observe_property("seeking", "bool", function(_, seeking)
  if seeking then
    if seek_held then
      begin_seek_hold()
    end
  else
    arm_resume()
  end
end)
mp.add_key_binding("F3", "fluid-toggle", toggle_fluid)

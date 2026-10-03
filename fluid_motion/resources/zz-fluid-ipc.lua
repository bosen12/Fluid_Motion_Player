-- Fluid Motion: named pipe. F3 only notifies the tray app — never injects vf.
local utils = require "mp.utils"
local pid = utils.getpid()
mp.set_property("options/input-ipc-server", "fluid-mpv-" .. pid)

local appdata = os.getenv("APPDATA") or ""
local alive_path = appdata .. "\\FluidMotion\\alive"
local hotkey_path = appdata .. "\\FluidMotion\\hotkey"
-- Per player. This was one shared file, so a seek here tore the filter off
-- every other connected mpv as well and made them all sit out the debounce.
local seek_hold_path = appdata .. "\\FluidMotion\\seek_hold-" .. pid
-- Quiet period after a seek settles before re-applying the filter. Short on
-- purpose: this is just debounce against still-dragging, not real work --
-- the actual rebuild cost is entirely in apply() afterward.
local SEEK_RESUME = 0.15
local seek_timer
local seek_held = false

-- The filter comes off for *every* seek, not just a drag.
--
-- An earlier version gated this on a second seek landing close behind the
-- first, on the reasoning that mpv reinitialises the VapourSynth script for
-- any seek anyway, so removing it first buys nothing. Total time to
-- interpolating does bear that out -- 0.69s to remove, seek and re-add versus
-- 0.71s to leave it loaded. But that is not the number anyone feels. What
-- gets felt is how long a seek takes to put a picture on screen, and there the
-- two are nothing alike: seeking with no VapourSynth in the chain shows a
-- picture in 0.14s, against 0.34s when the filter has to rebuild first.
--
-- Being 2.4x slower to respond is worse than a brief spell at source frame
-- rate, so the teardown is unconditional and the cost it used to carry -- the
-- wait before re-applying -- is addressed in the tray app instead, which now
-- wakes on the hold clearing rather than on its next poll.

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

-- Holding playback while the VapourSynth filter (re)initialises.
--
-- Building the RIFE pipeline -- loading an engine, or compiling one for a new
-- resolution (measured 38 s on the owner's machine) -- runs on mpv's core
-- thread, so mpv stops dead for that long. Left playing, audio ran on and
-- underran (2 underruns per 8 s block, measured) and long builds came back
-- with "Audio/Video desynchronisation". Paused first, the same build ran with
-- none. So playback is paused across every rebuild and resumed on the first
-- playback-restart after it, which mpv emits once the new pipeline has
-- produced a frame -- both when the filter is put on mid-file and for a new
-- file (both measured). This script still never adds the filter itself.
--
-- A vf change while paused makes mpv do a refresh seek. To begin_seek_hold
-- that looked like a user seek: it tore the filter straight back off, and
-- what followed was a seek every ~0.35 s, indefinitely, the filter never
-- staying on (measured against this script before the change). That was
-- already true of any apply that met a paused player; it is why seeks are
-- ignored while a hold is open.
--
-- Fluid Motion opens a hold through script-message "fluid-hold" before it
-- touches vf: "resume" when it paused a playing player for it, "keep" when
-- the user had paused and should stay paused. A new file with the filter
-- already on is held here, in on_preloaded, before mpv builds the pipeline.
local HOLD_SAFETY = 15
local hold = nil
local hold_timer

local function end_hold()
  if not hold then
    return
  end
  local resume = hold.resume
  hold = nil
  if hold_timer then
    hold_timer:kill()
    hold_timer = nil
  end
  if resume and mp.get_property_bool("pause") then
    mp.set_property_bool("pause", false)
  end
end

local function begin_hold(resume)
  hold = { resume = resume }
  if hold_timer then
    hold_timer:kill()
  end
  -- A lua timer fires on this script's thread even while the core is busy;
  -- the set_property inside then waits for the build to finish. So this only
  -- matters if no playback-restart ever comes.
  hold_timer = mp.add_timeout(HOLD_SAFETY, end_hold)
end

local function begin_seek_hold()
  if hold then
    return false  -- our own refresh seek, not the user's
  end
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

-- Tells Fluid Motion this script can hold playback across a rebuild. A player
-- still running an older copy of the script never sets it, and must then
-- never be paused for one: nothing in it would ever resume.
mp.set_property_native("user-data/fluid/hold-api", 1)
mp.register_script_message("fluid-hold", function(mode)
  begin_hold(mode == "resume")
end)
mp.add_hook("on_preloaded", 50, function()
  if hold or not fluid_alive() or not fluid_on() then
    return
  end
  if mp.get_property_bool("pause") then
    begin_hold(false)
  else
    mp.set_property_bool("pause", true)
    begin_hold(true)
  end
end)
mp.observe_property("pause", "bool", function(_, paused)
  -- Someone resumed mid-hold (the user, a script): theirs now, hands off.
  if hold and paused == false then
    hold = nil
    if hold_timer then
      hold_timer:kill()
      hold_timer = nil
    end
  end
end)

mp.add_timeout(0, strip_stale)
mp.register_event("start-file", strip_stale)
mp.register_event("file-loaded", strip_stale)
mp.register_event("seek", function()
  begin_seek_hold()
end)
mp.register_event("playback-restart", function()
  end_hold()
  arm_resume()
end)
mp.observe_property("seeking", "bool", function(_, seeking)
  if seeking then
    -- Property notifications are queued, not live. The seeking=true that a
    -- new file raises while a hold has mpv stopped arrived 4 ms *after* the
    -- hold resumed playback and tore the freshly built filter off (measured:
    -- 854x480 next file, 26 s build, then "vf remove @fluid" at once), which
    -- cost a second rebuild. Act only if mpv is still seeking now. A real
    -- seek is still caught: the "seek" event above drops the filter
    -- unconditionally (1f2f62a), this observer was only ever the backstop.
    if mp.get_property_bool("seeking") then
      begin_seek_hold()
    end
  else
    arm_resume()
  end
end)
-- Now that the file carries a pid, quitting mid-seek would leave one behind
-- for a pid that no longer exists. The watcher ignores anything older than
-- SEEK_HOLD_MAX_AGE, so it was never wrong -- just litter that accumulates.
mp.register_event("shutdown", clear_seek_hold)
mp.add_key_binding("F3", "fluid-toggle", toggle_fluid)

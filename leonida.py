#!/usr/bin/env python3
"""
Leonida -- the whole RC car, in one program, on the Pi.

Run it on the Raspberry Pi and nothing else anywhere:

    python3 leonida.py

Then open   http://<pi-ip>:8000   on a phone or laptop on the same network.

This replaces the old two-machine arrangement, where a laptop served the
joystick page and forwarded every command over the network to a motor API on
the Pi. There is no forwarding now: the process that draws the page is the
process holding the GPIO pins, so a stick movement reaches a motor in about a
millisecond instead of crossing the network twice.

What it serves:

    GET  /                      the joystick page
    GET  /ws                    WebSocket -- the normal control path
    GET  /drive?m1=&m2=&m3=     all three motors in one request (HTTP fallback)
    GET  /motor1?power=-100..100    one motor, kept for curl and old clients
    GET  /ping                  refreshes the watchdog on the fallback path
    GET  /stream/video          MJPEG from the webcam
    GET  /stream/audio          MP3 from the webcam's mic
    GET  /health                JSON status

Camera and mic come from a USB webcam via ffmpeg, started only while someone is
watching and shut down when the last viewer leaves. One capture process feeds
every viewer, so any number of people can watch at once.

Unlike the old motor API, this one has a deadman watchdog: if commands stop
arriving -- browser closed, phone asleep, out of Wi-Fi range -- the motors stop
by themselves within about half a second.
"""

import argparse
import array
import base64
import hashlib
import json
import math
import os
import re
import select
import signal
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# ---- config ---------------------------------------------------------------

PORT = 8000
FLIP_MOTOR3 = False        # set True if motor 3 steers the wrong way
MAX_BODY = 4096            # bytes; refuse a larger request body

DEADMAN = 0.6              # seconds of silence before the motors cut out
WATCHDOG_TICK = 0.1

START_WAIT = 20.0          # how long a viewer waits for a capture to come up
LINGER = 3.0               # keep a capture warm this long after the last viewer
VIDEO_QUEUE = 4            # frames held per browser before it starts skipping
AUDIO_QUEUE = 48           # chunks held per browser
MAX_FRAME_BUFFER = 4 * 1024 * 1024
BOUNDARY = "leonida"       # our multipart boundary

# ---- sound out of the car -------------------------------------------------

SPEAK_RATE = 16000         # Hz, mono, signed 16-bit little-endian throughout
SPEAK_CHUNK = SPEAK_RATE // 50            # 20ms of samples
SPEAK_BYTES = SPEAK_CHUNK * 2
SPEAK_QUEUE = 25           # ~half a second of speech held before dropping
SPEAK_LINGER = 2.0         # keep the speaker open this long after it goes quiet
HORN_TONES = (440.0, 550.0)               # a real car horn is two notes
HORN_FADE = 0.005          # seconds; stops a click when the button is released

# ---- depth overlay --------------------------------------------------------
#
# Optional, and entirely the browser's problem. The Pi does not do any of this:
# it holds the GPIO pins and has a half-second deadman on the motors, so
# spending its four cores on a neural network is a good way to make the car
# stutter. Measured on a laptop, one frame of Depth Anything V2 Small costs
# 25ms on WebGPU and 1250ms without it -- so the browser does the work, on a
# thread of its own, and the car never notices either way.
#
# The files are large and are fetched once with --fetch-model rather than
# living in git. They are served from the Pi, not a CDN, so the feature still
# works parked in a field with no internet.

ORT_VERSION = "1.27.0"
_ORT = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist"
_MODEL = ("https://huggingface.co/onnx-community/depth-anything-v2-small"
          "/resolve/main/onnx")

DEPTH_MODEL = "depth-fp16.onnx"
ASSET_SOURCES = [
    (f"{_ORT}/ort.webgpu.bundle.min.mjs", "ort.webgpu.bundle.min.mjs"),
    (f"{_ORT}/ort-wasm-simd-threaded.asyncify.mjs",
     "ort-wasm-simd-threaded.asyncify.mjs"),
    (f"{_ORT}/ort-wasm-simd-threaded.asyncify.wasm",
     "ort-wasm-simd-threaded.asyncify.wasm"),
    # fp16, not one of the quantized builds: WebGPU has no int8 kernels for
    # these ops and silently falls back to the CPU, which measured 60x slower.
    (f"{_MODEL}/model_fp16.onnx", DEPTH_MODEL),
]

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, "assets")
ASSET_TYPES = {".mjs": "text/javascript", ".js": "text/javascript",
               ".wasm": "application/wasm", ".onnx": "application/octet-stream"}

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
WS_MAX_PAYLOAD = 64 * 1024
WS_PING = 5.0              # silence before the server pings the browser
WS_QUIET = 4               # unanswered pings before we hang up (~20s)
WS_PATIENCE = 10.0         # limit on how long one frame may take to arrive

# socket.timeout only became an alias of the builtin TimeoutError in Python
# 3.10, and Pi OS ships 3.9 on Bullseye and 3.11 on Bookworm. Catching just one
# of them is the difference between a control socket that idles quietly and one
# that hangs up every five seconds on half the Raspberry Pis in the world.
TIMED_OUT = (TimeoutError, socket.timeout)

CFG = None                 # argparse namespace, filled in by main()

# ---- hardware -------------------------------------------------------------

try:
    from gpiozero import Motor

    motor1 = Motor(forward=12, backward=13, pwm=True)
    motor2 = Motor(forward=20, backward=21, pwm=True)
    motor3 = Motor(forward=18, backward=19, pwm=True)
    MOCK = False
except Exception as exc:   # not a Pi -- serve everything else anyway
    print(f"gpiozero unavailable ({exc}). Running in mock mode, no motors.")

    class MockMotor:
        def __init__(self):
            self.value = 0.0

        def stop(self):
            self.value = 0.0

    motor1, motor2, motor3 = MockMotor(), MockMotor(), MockMotor()
    MOCK = True

MOTORS = {1: motor1, 2: motor2, 3: motor3}

lock = threading.Lock()
_state = {1: 0.0, 2: 0.0, 3: 0.0}   # last commanded percent, per motor
_last_command = 0.0                 # monotonic clock of the last thing we heard
_last_log = 0.0


def coerce_power(raw):
    """Return (percent, None) or (None, error message).

    Everything that can reach a pin goes through here -- query strings, JSON
    bodies, and WebSocket messages alike. A stray "NaN" must not become a duty
    cycle.
    """
    if raw is None:
        return None, "missing required parameter: power"
    if isinstance(raw, bool):
        return None, "power must be a number between -100 and 100"
    try:
        percent = float(raw)
    except (TypeError, ValueError):
        return None, "power must be a number between -100 and 100"
    if percent != percent or percent in (float("inf"), float("-inf")):
        return None, "power must be a number between -100 and 100"
    if not -100 <= percent <= 100:
        return None, "power must be between -100 and 100"
    return percent, None


def _apply(index, percent):
    """Drive one motor. Caller holds `lock`."""
    value = percent / 100.0
    if index == 3 and FLIP_MOTOR3:
        value = -value
    motor = MOTORS[index]
    if value == 0:
        motor.stop()
    else:
        motor.value = max(-1.0, min(1.0, value))
    _state[index] = percent


def touch():
    """Note that we just heard from a controller. Feeds the watchdog."""
    global _last_command
    _last_command = time.monotonic()


def set_all(p1, p2, p3):
    """Set all three motors under a single lock -- this is the hot path."""
    changed = []
    with lock:
        for index, percent in ((1, p1), (2, p2), (3, p3)):
            if percent is None or _state[index] == percent:
                continue
            _apply(index, percent)
            changed.append((index, percent))
    touch()
    if changed:
        _log(changed)
    return changed


def set_one(index, percent):
    with lock:
        changed = _state[index] != percent
        if changed:
            _apply(index, percent)
    touch()
    if changed:
        _log([(index, percent)])


def _log(changed):
    """Throttled console echo.

    The joystick can produce updates faster than a terminal can scroll, and on
    the Pi every line is a write to the journal. A stop is always shown; the
    rest are sampled.
    """
    global _last_log
    now = time.monotonic()
    if any(p != 0 for _, p in changed) and now - _last_log < 0.1:
        return
    _last_log = now
    print("  " + "  ".join(f"motor{i} -> {p:g}%" for i, p in changed))


def stop_all(reason=None):
    """Everything off: wheels and horn both.

    A horn stuck on is arguably worse than a motor stuck on -- you cannot hear
    yourself think while you go and fix it -- so it stops wherever the motors
    stop, which means the STOP button and the space bar silence it too.
    """
    with lock:
        moving = any(_state.values())
        for index, motor in MOTORS.items():
            motor.stop()
            _state[index] = 0.0
    sounding = SPEAKER.horn
    SPEAKER.set_horn(False)
    if reason and (moving or sounding):
        print(f"  ** {reason} -- motors stopped")
    return moving or sounding


def watchdog():
    """Cuts the motors when we stop hearing from whoever is driving.

    The old motor API held the last value forever, which is fine on a bench and
    alarming on a floor: close the tab mid-throttle and the car keeps going.
    The page sends a keepalive several times a second, so silence really does
    mean nobody is there.
    """
    while True:
        time.sleep(WATCHDOG_TICK)
        with lock:
            moving = any(_state.values())
        # The horn is held down from the page the same way the throttle is, so
        # it needs the same deadman: lose the phone mid-blast and it stops.
        if not moving and not SPEAKER.horn:
            continue
        if time.monotonic() - _last_command > DEADMAN:
            stop_all(f"no command for {DEADMAN:.1f}s")


# ---- stream fan-out -------------------------------------------------------
#
# One ffmpeg per feed, however many browsers are watching. The capture starts
# when the first viewer subscribes and is killed after the last one leaves, so
# the webcam is only powered while someone is actually looking at it.


class Subscriber:
    """One browser's queue. A slow viewer drops data, never the others."""

    def __init__(self, limit):
        self.limit = limit
        self.items = deque()
        self.cv = threading.Condition()
        self.ended = False

    def put(self, item):
        with self.cv:
            self.items.append(item)
            while len(self.items) > self.limit:
                self.items.popleft()   # fall behind and you skip ahead, not lag
            self.cv.notify()

    def get(self, timeout):
        """Next item, or None if nothing arrived in time (check .ended)."""
        with self.cv:
            if not self.items and not self.ended:
                self.cv.wait(timeout)
            return self.items.popleft() if self.items else None

    def end(self):
        with self.cv:
            self.ended = True
            self.cv.notify()


class _Attempt:
    """One capture process, and the viewers sharing it."""

    def __init__(self):
        self.ready = threading.Event()
        self.subs = set()
        self.live = False
        self.error = None      # (status, body, content-type)
        self.ctype = None


class Stream:
    """A feed: an ffmpeg command line, fanned out to every browser."""

    def __init__(self, name, argv, split, wrap, queue_limit, ctype):
        self.name = name
        self.argv = argv               # callable -> list of ffmpeg arguments
        self.split = split             # callable -> chunk -> [item, ...]
        self.wrap = wrap               # callable -> item -> bytes for a browser
        self.queue_limit = queue_limit
        self.ctype = ctype
        self.lock = threading.Lock()
        self.thread = None
        self.attempt = None

    def subscribe(self):
        """(attempt, subscriber, None), or (None, None, error)."""
        sub = Subscriber(self.queue_limit)
        with self.lock:
            if self.thread is None:
                self.attempt = _Attempt()
                self.attempt.subs.add(sub)     # before the pump can look
                self.thread = threading.Thread(
                    target=self._pump, args=(self.attempt,), daemon=True)
                self.thread.start()
            else:
                self.attempt.subs.add(sub)
            att = self.attempt

        if not att.ready.wait(START_WAIT):
            self.unsubscribe(att, sub)
            return None, None, (504, b'{"error":"the capture did not start"}',
                                "application/json")
        if not att.live:
            self.unsubscribe(att, sub)
            return None, None, att.error or (
                502, b'{"error":"capture unavailable"}', "application/json")
        return att, sub, None

    def unsubscribe(self, att, sub):
        with self.lock:
            att.subs.discard(sub)

    def viewers(self):
        with self.lock:
            return len(self.attempt.subs) if self.attempt else 0

    def _broadcast(self, att, item):
        with self.lock:
            subs = list(att.subs)
        for sub in subs:
            sub.put(item)

    def _fail(self, att, message):
        att.error = (502, json.dumps({"error": message}).encode(),
                     "application/json")

    def _pump(self, att):
        proc = None
        errors = []
        try:
            argv = self.argv()
            try:
                proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            except OSError as e:
                self._fail(att, f"cannot run ffmpeg: {e}")
                return
            drain = threading.Thread(target=_drain, args=(proc.stderr, errors),
                                     daemon=True)
            drain.start()

            # Wait for real bytes before telling the browser this worked. A
            # missing device makes ffmpeg exit immediately, and we would rather
            # send its complaint than an empty 200 the page cannot explain.
            chunk = proc.stdout.read1(65536)
            if not chunk:
                proc.wait(timeout=2)
                drain.join(timeout=1)
                self._fail(att, _why(errors, self.name))
                return

            att.ctype = self.ctype
            att.live = True
            att.ready.set()
            print(f"  [{self.name}] capture up")

            split = self.split()
            empty_since = None
            while chunk:
                for item in split(chunk):
                    self._broadcast(att, item)

                # Hang on briefly after the last viewer leaves rather than
                # tearing the camera down the instant nobody is subscribed.
                # A browser reconnecting, or the snapshot fallback polling one
                # frame at a time, would otherwise restart ffmpeg constantly --
                # and a webcam takes far longer to open than it does to serve.
                with self.lock:
                    if att.subs:
                        empty_since = None
                    elif empty_since is None:
                        empty_since = time.monotonic()
                    elif time.monotonic() - empty_since > LINGER:
                        self.thread = None
                        return

                # read1 hands over whatever has arrived rather than waiting to
                # fill a buffer -- that is what keeps the feed live.
                chunk = proc.stdout.read1(65536)
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            if not att.live:
                self._fail(att, f"{self.name} capture failed: {e}")
        finally:
            if proc is not None:
                _kill(proc)
                if att.live:
                    print(f"  [{self.name}] capture down")
            att.ready.set()                       # release anyone still waiting
            with self.lock:
                if self.thread is threading.current_thread():
                    self.thread = None
                subs = list(att.subs)
            for sub in subs:
                sub.end()


def _drain(pipe, sink):
    """Keep ffmpeg's stderr moving, and keep the last of it for error messages.

    An undrained stderr pipe fills and blocks ffmpeg mid-frame, which looks
    exactly like a camera that has frozen.
    """
    try:
        for line in pipe:
            text = line.decode("utf-8", "replace").strip()
            if text:
                sink.append(text)
                del sink[:-10]
    except (OSError, ValueError):
        pass


def _why(errors, name):
    """ffmpeg's own last words, if it left any."""
    for text in reversed(errors):
        if "Immediate exit" not in text:
            return text
    return f"{name} produced no data -- is the device connected?"


def _kill(proc):
    try:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    except (OSError, subprocess.SubprocessError):
        pass
    # Any of these may be None -- the speaker sends its stdout to DEVNULL --
    # and this runs from a finally block, where an exception would skip the
    # cleanup that follows it.
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is None:
            continue
        try:
            pipe.close()
        except (OSError, ValueError):
            pass


# ---- the speaker ----------------------------------------------------------
#
# One ffmpeg writing to the USB audio dongle, fed by a thread that builds 20ms
# chunks out of whatever is making noise: the horn, speech arriving from a
# phone, or both at once. Started when something wants to be heard and stopped
# a couple of seconds after the car goes quiet, so the device is not held open
# all day but a second horn press is still instant.


def speaker_argv():
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-fflags", "nobuffer",
            "-f", "s16le", "-ar", str(SPEAK_RATE), "-ac", "1", "-i", "-"]
    if CFG.speaker_format == "wav":        # writing a file, for testing
        argv += ["-y", "-f", "wav", CFG.speaker_device]
    else:
        argv += ["-f", CFG.speaker_format, CFG.speaker_device]
    return argv


def horn_wave(tones):
    """Exactly one second of the two-tone horn.

    A whole second of whole-numbered frequencies contains a whole number of
    cycles of each, so playing this buffer end to end forever is seamless --
    no click at the loop point, and no arithmetic at all while it sounds.
    """
    n = SPEAK_RATE
    wave = array.array("h", bytes(2 * n))
    amp = 0.32                             # two tones summed, with headroom
    for i in range(n):
        t = i / SPEAK_RATE
        v = sum(math.sin(2 * math.pi * f * t) for f in tones) * amp
        wave[i] = int(max(-1.0, min(1.0, v)) * 32767)
    return wave.tobytes()


SILENCE = bytes(SPEAK_BYTES)
HORN_WAVE = b""            # built in main(), once CFG.horn_tones is known


def mix(a, b):
    """Sum two equal-length S16LE buffers, clipping rather than wrapping.

    Only reached when the horn and someone talking overlap; a single source is
    passed through untouched. Deliberately not audioop, which was removed in
    Python 3.13 and would strand this on older interpreters only.
    """
    xa = array.array("h")
    xa.frombytes(a)
    xb = array.array("h")
    xb.frombytes(b)
    for i in range(len(xa)):
        v = xa[i] + xb[i]
        xa[i] = 32767 if v > 32767 else (-32768 if v < -32768 else v)
    return xa.tobytes()


class Speaker:
    """Everything the car says out loud."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.horn = False
        self.at = 0                        # position in the horn loop, samples
        self.fade = 0                      # samples of fade-out still owed
        self.buf = bytearray()             # speech waiting to be played
        self.primed = False                # jitter buffer has enough to start
        self.last_speech = 0.0
        self.error = None

    # ---- what the rest of the program calls ----

    def set_horn(self, on):
        with self.lock:
            if on == self.horn:
                return
            self.horn = on
            if not on:
                # Cutting a sine mid-cycle is a click. Ramp it out instead.
                self.fade = int(HORN_FADE * SPEAK_RATE)
        if on:
            self._ensure()

    def speak(self, pcm):
        with self.lock:
            self.buf.extend(pcm)
            limit = SPEAK_QUEUE * SPEAK_BYTES
            if len(self.buf) > limit:      # a slow link falls behind, not lags
                del self.buf[:len(self.buf) - limit]
            if len(self.buf) >= 2 * SPEAK_BYTES:
                self.primed = True
            self.last_speech = time.monotonic()
        self._ensure()

    def hush(self):
        """Everything off, right now."""
        with self.lock:
            self.horn = False
            self.fade = 0
            self.buf.clear()
            self.primed = False

    def status(self):
        with self.lock:
            return {"horn": self.horn,
                    "speaking": bool(self.buf),
                    "live": self.thread is not None,
                    "device": CFG.speaker_device if CFG else None,
                    "error": self.error}

    # ---- the pipeline ----

    def _ensure(self):
        with self.lock:
            if self.thread is not None:
                return
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _next(self):
        """The next 20ms to play, or None if there is nothing to say."""
        with self.lock:
            horn, fade = self.horn, self.fade
            speech = None
            if self.primed and len(self.buf) >= SPEAK_BYTES:
                speech = bytes(self.buf[:SPEAK_BYTES])
                del self.buf[:SPEAK_BYTES]
                if not self.buf:
                    self.primed = False    # wait for a cushion before resuming
            elif self.buf and time.monotonic() - self.last_speech > 0.2:
                speech = bytes(self.buf).ljust(SPEAK_BYTES, b"\0")   # last words
                self.buf.clear()
                self.primed = False

            tone = None
            if horn or fade:
                tone = self._tone(fade)

        if tone is None:
            return speech
        if speech is None:
            return tone
        return mix(tone, speech)

    def _tone(self, fade):
        """One chunk of horn. Caller holds the lock."""
        start = self.at
        chunk = HORN_WAVE[start * 2:(start + SPEAK_CHUNK) * 2]
        self.at = (start + SPEAK_CHUNK) % SPEAK_RATE

        if not fade:
            return chunk
        # Ramp the last few milliseconds down to nothing, then go quiet.
        out = array.array("h")
        out.frombytes(chunk)
        for i in range(len(out)):
            if i < fade:
                out[i] = int(out[i] * (1 - i / fade))
            else:
                out[i] = 0
        self.fade = 0
        return out.tobytes()

    def _run(self):
        proc = None
        errors = []
        try:
            proc = subprocess.Popen(speaker_argv(), stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
            threading.Thread(target=_drain, args=(proc.stderr, errors),
                             daemon=True).start()
            print("  [speaker] on")
            self.error = None

            # Paced by the clock rather than by the device blocking, so that
            # pointing this at a file for testing produces audio of the right
            # length instead of everything at once.
            quiet_since = None
            deadline = time.monotonic()
            while True:
                chunk = self._next()
                if chunk is None:
                    now = time.monotonic()
                    if quiet_since is None:
                        quiet_since = now
                    elif now - quiet_since > SPEAK_LINGER:
                        break
                    chunk = SILENCE
                else:
                    quiet_since = None

                proc.stdin.write(chunk)
                proc.stdin.flush()

                deadline += SPEAK_CHUNK / SPEAK_RATE
                nap = deadline - time.monotonic()
                if nap > 0:
                    time.sleep(nap)
                else:
                    deadline = time.monotonic()      # fell behind; resync
        except (OSError, ValueError) as e:
            self.error = f"speaker failed: {e}"
            if errors:
                self.error = errors[-1]
            print(f"  [speaker] {self.error}")
        finally:
            # Whatever went wrong above, the thread handle has to be cleared or
            # the speaker can never be started again.
            try:
                if proc is not None:
                    # Closing stdin is an end-of-input, which lets ffmpeg finish
                    # of its own accord. That matters when the output is a file:
                    # killed mid-write, a WAV never gets its header and is
                    # unreadable. Only force it if it will not go quietly.
                    if proc.stdin is not None:
                        try:
                            proc.stdin.close()
                        except OSError:
                            pass
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
                    _kill(proc)
                    print("  [speaker] off")
            except Exception as e:                    # never leave it wedged
                print(f"  [speaker] shutdown: {e}")
            with self.lock:
                self.thread = None


SPEAKER = Speaker()


# ---- framing --------------------------------------------------------------

# Markers that stand alone: no length field, nothing to skip over.
_STANDALONE = {0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7}

# The Huffman tables from the JPEG specification, Annex K.
#
# Most USB webcams emit "MJPEG" frames with no Huffman table in them at all --
# the AVI1 convention, where every frame is understood to use these standard
# tables and leaves them out to save a couple of hundred bytes per frame. A
# decoder that expects a whole JPEG file has nothing to decode with. ffmpeg
# copes, which is why the stream looks healthy from the Pi's side; browsers do
# not. Safari refuses the image outright and shows a broken-image icon, and
# Chrome renders convincing garbage, which is arguably worse.
#
# So when a frame arrives without tables we put these back. Written out rather
# than copied from a sample file on purpose: ffmpeg's encoder emits *optimised*
# tables tuned to one image, and those are precisely the wrong thing to hand a
# frame that was encoded against the standard ones.
_DC_LUMA_BITS = (0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
_DC_CHROMA_BITS = (0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
_DC_VALUES = tuple(range(12))

_AC_LUMA_BITS = (0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D)
_AC_LUMA_VALUES = (
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
)

_AC_CHROMA_BITS = (0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77)
_AC_CHROMA_VALUES = (
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
    0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
    0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
)


def _dht_segment():
    """The four standard tables as one DHT segment, ready to splice in."""
    body = b""
    for slot, bits, values in (
            (0x00, _DC_LUMA_BITS, _DC_VALUES),        # DC, table 0: luminance
            (0x10, _AC_LUMA_BITS, _AC_LUMA_VALUES),   # AC, table 0
            (0x01, _DC_CHROMA_BITS, _DC_VALUES),      # DC, table 1: chrominance
            (0x11, _AC_CHROMA_BITS, _AC_CHROMA_VALUES),   # AC, table 1
    ):
        assert sum(bits) == len(values), "Huffman table is inconsistent"
        body += bytes((slot,)) + bytes(bits) + bytes(values)
    return b"\xff\xc4" + struct.pack(">H", len(body) + 2) + body


STD_DHT = _dht_segment()
_said_no_tables = False


def jpeg_frames():
    """Cuts ffmpeg's MJPEG output into whole JPEG images.

    ffmpeg hands us one image after another with nothing in between, and a
    browser joining halfway through must be given whole frames or it renders
    garbage -- so we have to find the boundaries ourselves.

    The obvious way, scanning for the end-of-image marker FFD9, is wrong: a
    webcam frame often carries an EXIF thumbnail, and that thumbnail is itself
    a JPEG whose own FFD9 arrives long before the real image ends. Cutting
    there yields a truncated frame the browser refuses to decode, and the feed
    stutters or dies.

    So we walk the file structure properly instead. Every segment except the
    entropy-coded scan declares its length, so APP1 (where the thumbnail
    lives) is stepped over whole. Only after SOS do we scan byte by byte, and
    there we ignore stuffed FF00 and the restart markers, which are part of
    the image data rather than the end of it.
    """
    buf = bytearray()          # begins at the current frame's SOI once synced
    pos = 0                    # how far into buf we have parsed
    synced = False
    scan = False               # inside entropy-coded data after SOS
    has_dht = False            # did this frame bring its own Huffman tables?
    sos_at = -1                # where to splice them in if it did not

    def feed(chunk):
        nonlocal pos, synced, scan, has_dht, sos_at
        global _said_no_tables
        buf.extend(chunk)
        out = []

        while True:
            if not synced:
                i = buf.find(b"\xff\xd8\xff")
                if i < 0:
                    if len(buf) > 2:
                        del buf[:len(buf) - 2]     # keep a possible partial SOI
                    break
                del buf[:i]                        # drop anything before it
                synced, scan, pos = True, False, 2
                has_dht, sos_at = False, -1

            n = len(buf)
            end = -1                               # index just past this EOI
            while True:
                if scan:
                    i = buf.find(0xFF, pos)
                    if i < 0:
                        pos = n
                        break
                    if i + 1 >= n:
                        pos = i                    # keep the FF for next time
                        break
                    b = buf[i + 1]
                    if b == 0xFF:
                        pos = i + 1                # fill byte
                        continue
                    if b == 0x00 or 0xD0 <= b <= 0xD7:
                        pos = i + 2                # stuffing or restart
                        continue
                    scan, pos = False, i           # a real marker
                    continue

                if pos + 1 >= n:
                    break
                if buf[pos] != 0xFF:               # lost the thread
                    synced = False
                    del buf[:2]
                    break
                marker = buf[pos + 1]
                if marker == 0xD9:                 # end of image
                    end = pos + 2
                    break
                if marker == 0xFF:
                    pos += 1
                    continue
                if marker in _STANDALONE:
                    pos += 2
                    continue
                if pos + 3 >= n:
                    break
                length = (buf[pos + 2] << 8) | buf[pos + 3]
                if length < 2:                     # malformed
                    synced = False
                    del buf[:2]
                    break
                if marker == 0xC4:                 # this frame has its own
                    has_dht = True
                elif marker == 0xDA:               # tables must precede the scan
                    sos_at = pos
                pos += 2 + length
                if marker == 0xDA:                 # start of scan
                    scan = True

            if end < 0:
                if not synced:
                    continue                       # resynced -- go round again
                if len(buf) > MAX_FRAME_BUFFER:
                    buf.clear()
                    synced, scan, pos = False, False, 0
                break                              # wait for more bytes

            if has_dht or sos_at < 0:
                out.append(bytes(buf[:end]))
            else:
                # A tableless webcam frame. Give it the standard tables it was
                # encoded against, so what leaves here is a whole JPEG file
                # rather than one that only ffmpeg is forgiving enough to read.
                out.append(bytes(buf[:sos_at]) + STD_DHT +
                           bytes(buf[sos_at:end]))
                if not _said_no_tables:
                    _said_no_tables = True
                    print("  [video] camera sends frames without Huffman "
                          "tables -- adding the standard ones")
            del buf[:end]
            synced, scan, pos = False, False, 0
            has_dht, sos_at = False, -1

        return out

    return feed


def audio_chunks():
    """MP3 needs no splitting -- decoders resync on their own."""
    return lambda chunk: [chunk] if chunk else []


def mjpeg_wrapper():
    """Frame -> one multipart part, addressed to this browser."""
    head = ("--" + BOUNDARY + "\r\nContent-Type: image/jpeg\r\n"
            "Content-Length: %d\r\n\r\n")

    def wrap(frame):
        return (head % len(frame)).encode("ascii") + frame + b"\r\n"

    return wrap


def mp3_wrapper():
    """Drops the partial MP3 frame a late joiner would otherwise start on."""
    synced = False

    def wrap(chunk):
        nonlocal synced
        if synced:
            return chunk
        for i in range(len(chunk) - 1):    # MP3 frames open with 11 set bits
            if chunk[i] == 0xFF and (chunk[i + 1] & 0xE0) == 0xE0:
                synced = True
                return chunk[i:]
        return b""

    return wrap


# ---- capture command lines ------------------------------------------------


def video_argv():
    """ffmpeg reading the webcam.

    On Linux the webcam already produces MJPEG, so we copy it through
    untouched: no transcode, almost no CPU, and latency is whatever the USB
    bus costs. Elsewhere (a Mac, for testing) the input is raw and we encode.
    """
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-fflags", "nobuffer", "-flags", "low_delay"]
    fmt = CFG.video_format
    argv += ["-f", fmt]
    if fmt == "v4l2":
        argv += ["-input_format", "mjpeg",
                 "-video_size", f"{CFG.width}x{CFG.height}",
                 "-framerate", str(CFG.fps)]
    argv += ["-i", CFG.video_device]
    if fmt == "v4l2":
        argv += ["-c:v", "copy"]
    else:
        argv += ["-c:v", "mjpeg", "-q:v", "6",
                 "-s", f"{CFG.width}x{CFG.height}", "-r", str(CFG.fps)]
    argv += ["-f", "mjpeg", "-"]
    return argv


def audio_argv():
    """ffmpeg reading the mic, encoding MP3 -- the one format every browser
    will play from a live stream, iPhones included."""
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-fflags", "nobuffer",
            "-f", CFG.audio_format, "-i", CFG.audio_device,
            "-ac", "1", "-ar", "44100",
            "-c:a", "libmp3lame", "-b:a", "64k", "-reservoir", "0",
            "-flush_packets", "1", "-f", "mp3", "-"]


def detect_audio_device():
    """The webcam's ALSA capture device, as plughw:<card>,<device>.

    A USB webcam lands on an unpredictable card number, and guessing wrong
    costs an afternoon, so ask rather than hardcode. The speaker's USB dongle
    also enumerates a capture side -- a mic input with nothing plugged into it
    -- so prefer anything that names itself a camera over just taking the first
    card, which is silence if the dongle happens to enumerate first.
    """
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return "default"
    cards = re.findall(r"^card (\d+): (.*?device (\d+):.*)$", out, re.M)
    if not cards:
        return "default"

    def rank(entry):
        line = entry[1].lower()
        if "cam" in line:                     # Camera, Webcam, USB Camera
            return 0
        if "usb audio" in line:               # the speaker dongle's mic input
            return 2
        return 1

    num, _line, dev = min(cards, key=rank)
    return f"plughw:{num},{dev}"


def depth_ready():
    """Are all the depth-overlay assets on disk?"""
    return all(os.path.isfile(os.path.join(ASSET_DIR, name))
               for _url, name in ASSET_SOURCES)


def fetch_assets():
    """Download the runtime and the model onto the Pi, once."""
    import urllib.request

    os.makedirs(ASSET_DIR, exist_ok=True)
    print(f"\n  fetching depth-overlay assets into {ASSET_DIR}")
    for url, name in ASSET_SOURCES:
        target = os.path.join(ASSET_DIR, name)
        if os.path.isfile(target):
            print(f"    have  {name}")
            continue
        part = target + ".part"
        print(f"    get   {name} ... ", end="", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "leonida"})
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(part, "wb") as out:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(262144)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r    get   {name} ... "
                              f"{done * 100 // total}%", end="", flush=True)
            os.replace(part, target)
            print(f"\r    got   {name} ({done / 1e6:.1f} MB)      ")
        except Exception as e:                    # network, disk, anything
            if os.path.exists(part):
                os.remove(part)
            print(f"\n    FAILED: {e}")
            print("\n  Nothing else was changed. Fix the problem and run it "
                  "again;\n  files already downloaded are kept.\n")
            return 1
    print("\n  Done. Turn the overlay on under Settings in the page.\n")
    return 0


def describe_jpeg(frame):
    """What a decoder will make of this frame: its segments and its size."""
    names = {0xC0: "SOF0 baseline", 0xC1: "SOF1", 0xC2: "SOF2 progressive",
             0xC4: "DHT huffman", 0xDB: "DQT quant", 0xDD: "DRI restart",
             0xDA: "SOS scan", 0xE0: "APP0/JFIF", 0xE1: "APP1/EXIF",
             0xFE: "COM"}
    found, size, i = [], None, 2
    while i + 1 < len(frame) and frame[i] == 0xFF:
        marker = frame[i + 1]
        found.append(names.get(marker, hex(marker)))
        if marker == 0xDA:
            break
        length = (frame[i + 2] << 8) | frame[i + 3]
        if marker in (0xC0, 0xC1, 0xC2):
            size = ((frame[i + 7] << 8) | frame[i + 8],
                    (frame[i + 5] << 8) | frame[i + 6])
        i += 2 + length
    return found, size


def check_camera():
    """Grab one frame and say whether it is something a browser can show.

    The camera can look perfectly healthy from the Pi -- ffmpeg is forgiving
    about frames a browser will not touch -- so this reports what is actually
    in a frame rather than merely whether one arrived.
    """
    stream = Stream("video", video_argv, jpeg_frames, mjpeg_wrapper,
                    VIDEO_QUEUE, "image/jpeg")
    print(f"\n  opening {CFG.video_device} via {CFG.video_format} ...")
    att, sub, error = stream.subscribe()
    if error is not None:
        print(f"  FAILED: {json.loads(error[1])['error']}\n")
        return 1
    try:
        frame = None
        deadline = time.monotonic() + START_WAIT
        while frame is None and time.monotonic() < deadline:
            frame = sub.get(1.0)
            if sub.ended:
                break
    finally:
        stream.unsubscribe(att, sub)

    if frame is None:
        print("  FAILED: the camera opened but sent no frame\n")
        return 1

    found, size = describe_jpeg(frame)
    path = "/tmp/leonida-frame.jpg"
    with open(path, "wb") as f:
        f.write(frame)

    print(f"  got a frame: {len(frame)} bytes"
          f"{', %dx%d' % size if size else ''}")
    print(f"  segments:    {', '.join(found)}")
    print(f"  saved to     {path}")
    if _said_no_tables:
        print("\n  This camera sends frames with no Huffman tables, which is\n"
              "  normal for USB webcams. The standard ones are being added,\n"
              "  so what reaches the browser is a complete JPEG.")
    if any(s.startswith("SOF2") for s in found):
        print("\n  WARNING: progressive JPEG. Some browsers will not show this\n"
              "  as a live feed. Try a different --width/--height.")
    print("\n  Open that file, or copy it off the Pi and open it. If it looks\n"
          "  right, the camera and the framing are fine and the problem is\n"
          "  in the browser; if it looks wrong, it is the capture.\n")
    return 0


def detect_speaker_device():
    """The ALSA playback device the speaker is plugged into.

    The speaker hangs off a USB-to-3.5mm dongle, so a USB audio card is what we
    want; the Pi's own headphone jack and HDMI are both wrong now, and HDMI is
    especially confusing because a Pi with a monitor attached enumerates it
    first and the horn comes out of the monitor.
    """
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return "default"
    # The whole line, not just the short name: a dongle is usually
    # "card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]",
    # where the only word that identifies it is in the bracketed long name.
    cards = re.findall(r"^card (\d+): (.*?device (\d+):.*)$", out, re.M)
    if not cards:
        return "default"

    def rank(entry):
        line = entry[1].lower()
        if "usb" in line:                              # the dongle
            return 0
        if "headphone" in line or "audiojack" in line:  # the Pi's own jack
            return 1
        if "hdmi" in line or "vc4" in line:            # the monitor
            return 3
        return 2

    num, _line, dev = min(cards, key=rank)
    return f"plughw:{num},{dev}"


def list_devices():
    for label, argv in (("cameras", ["v4l2-ctl", "--list-devices"]),
                        ("microphones", ["arecord", "-l"]),
                        ("speakers", ["aplay", "-l"])):
        print(f"\n--- {label} ---")
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=10)
            print((r.stdout or r.stderr).strip() or "(nothing reported)")
        except FileNotFoundError:
            print(f"{argv[0]} not installed")
        except (OSError, subprocess.SubprocessError) as e:
            print(f"could not run {argv[0]}: {e}")
    print()


STREAMS = {}   # filled in by main(), once CFG exists


# ---- websocket ------------------------------------------------------------
#
# Hand-rolled, because it is about eighty lines and saves depending on a
# package that is not on a fresh Pi OS image. Only what a browser actually
# sends is supported: small, unfragmented text frames, plus ping and close.


def ws_accept(key):
    digest = hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


class WSConn:
    """Frames over the hijacked socket.

    Deliberately does not use the handler's rfile. That is a BufferedReader,
    and a timeout taken while it holds a partial frame can lose the bytes it
    had already collected. Owning the buffer here means a timeout can only
    happen at a frame boundary, which is what makes it safe to stop waiting,
    send a ping, and carry on reading the same stream.

    (This assumes the browser sends nothing before it has seen our 101, which
    is what every browser does -- a handshake it has not read cannot be one it
    is already replying to.)
    """

    def __init__(self, sock):
        self.sock = sock
        self.buf = bytearray()

    def _fill(self, timeout):
        self.sock.settimeout(timeout)
        chunk = self.sock.recv(65536)
        if not chunk:
            raise ConnectionResetError("peer closed the control socket")
        self.buf.extend(chunk)

    def _take(self, n):
        while len(self.buf) < n:
            self._fill(WS_PATIENCE)
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def read(self, idle):
        """(opcode, payload).

        Raises TimeoutError if nothing at all arrived within `idle` -- a safe
        point, because there is no half-read frame outstanding.
        """
        if not self.buf:
            self._fill(idle)
        head = self._take(2)
        opcode = head[0] & 0x0F
        masked = head[1] & 0x80
        length = head[1] & 0x7F

        if length == 126:
            length = struct.unpack(">H", self._take(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._take(8))[0]

        # A browser always masks, and never sends anything remotely this big.
        if not masked or length > WS_MAX_PAYLOAD:
            raise ConnectionResetError("malformed frame")

        mask = self._take(4)
        payload = self._take(length) if length else b""
        if length:
            payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        return opcode, payload

    def write(self, payload, opcode=0x1):
        n = len(payload)
        if n < 126:
            head = struct.pack(">BB", 0x80 | opcode, n)
        elif n < 65536:
            head = struct.pack(">BBH", 0x80 | opcode, 126, n)
        else:
            head = struct.pack(">BBQ", 0x80 | opcode, 127, n)
        self.sock.settimeout(WS_PATIENCE)
        self.sock.sendall(head + payload)


class Controllers:
    """How many browsers currently hold a control socket."""

    def __init__(self):
        self.n = 0
        self.lock = threading.Lock()

    def join(self):
        with self.lock:
            self.n += 1
            return self.n

    def leave(self):
        with self.lock:
            self.n = max(0, self.n - 1)
            return self.n

    def count(self):
        with self.lock:
            return self.n


CONTROLLERS = Controllers()


def drive_from_text(message):
    """A control message: "<m1>,<m2>,<m3>". True if it was understood."""
    parts = message.split(",")
    if len(parts) != 3:
        return False
    powers = []
    for part in parts:
        percent, error = coerce_power(part.strip())
        if error is not None:
            return False
        powers.append(percent)
    set_all(*powers)
    return True


# ---- http -----------------------------------------------------------------


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Leonida</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --line:#2a3138; --ink:#e6edf3;
    --muted:#8b949e; --accent:#3fb950; --accent2:#58a6ff; --danger:#f85149;
  }
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  html,body{margin:0;height:100%;overflow:hidden}
  body{
    background:#000; color:var(--ink);
    font:15px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    overscroll-behavior:none; touch-action:none;
  }

  /* ---- camera feed: the whole window ---- */
  #feed{
    position:fixed; inset:0; width:100%; height:100%;
    object-fit:cover; background:#000; z-index:0; display:block;
  }
  #feed.dead{opacity:0}
  /* depth sits directly on the feed, under everything else */
  #depth{
    position:fixed; inset:0; width:100%; height:100%;
    object-fit:cover; z-index:1; display:none; pointer-events:none;
  }
  #depth.on{display:block}
  #feedmsg{
    position:fixed; inset:0; z-index:2; display:none;
    align-items:center; justify-content:center; flex-direction:column; gap:8px;
    color:var(--muted); font-size:14px; text-align:center; padding:24px;
  }
  #feedmsg.show{display:flex}
  #feedmsg .spin{width:26px;height:26px;border-radius:50%;
    border:2px solid var(--line);border-top-color:var(--accent2);
    animation:spin 1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* ---- everything else floats on top of it ---- */
  .hud{position:fixed;inset:0;z-index:3;pointer-events:none}
  .hud>*{pointer-events:auto}

  .top{
    position:absolute;top:0;left:0;right:0;display:flex;align-items:center;gap:8px;
    padding:calc(10px + env(safe-area-inset-top)) calc(12px + env(safe-area-inset-right))
            10px calc(12px + env(safe-area-inset-left));
    background:linear-gradient(180deg,rgba(0,0,0,.65),rgba(0,0,0,0));
  }
  .top h1{font-size:15px;margin:0;font-weight:600;flex:1;
          text-shadow:0 1px 3px rgba(0,0,0,.8)}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--danger);
       box-shadow:0 0 8px var(--danger);transition:.2s;flex:none}
  .dot.ok{background:var(--accent);box-shadow:0 0 8px var(--accent)}

  .btn{
    border:1px solid rgba(255,255,255,.18); background:rgba(13,17,23,.6);
    -webkit-backdrop-filter:blur(8px); backdrop-filter:blur(8px);
    color:var(--ink); border-radius:10px; padding:8px 11px; font-size:14px;
    font-weight:600; cursor:pointer; line-height:1; white-space:nowrap;
  }
  .btn.on{border-color:var(--accent2);color:var(--accent2)}

  /* joystick, bottom-right so a thumb reaches it in landscape */
  .padwrap{
    position:absolute; right:calc(16px + env(safe-area-inset-right));
    bottom:calc(16px + env(safe-area-inset-bottom));
    display:flex; flex-direction:column; align-items:center; gap:10px;
  }
  #pad{
    position:relative; width:min(46vh,44vw,300px); height:min(46vh,44vw,300px);
    border-radius:50%;
    background:radial-gradient(circle at 50% 50%, rgba(27,35,44,.55) 0%, rgba(10,14,18,.7) 70%);
    border:1px solid rgba(255,255,255,.18); touch-action:none;
    -webkit-backdrop-filter:blur(6px); backdrop-filter:blur(6px);
    box-shadow:inset 0 0 40px rgba(0,0,0,.6), 0 4px 24px rgba(0,0,0,.5);
  }
  #pad::before,#pad::after{content:"";position:absolute;background:#fff;opacity:.22}
  #pad::before{left:50%;top:8%;bottom:8%;width:1px;transform:translateX(-.5px)}
  #pad::after{top:50%;left:8%;right:8%;height:1px;transform:translateY(-.5px)}
  #knob{
    position:absolute;width:34%;height:34%;left:33%;top:33%;border-radius:50%;
    background:radial-gradient(circle at 35% 30%, var(--accent2), #1f6feb);
    box-shadow:0 6px 18px rgba(0,0,0,.6);will-change:transform;touch-action:none;
  }

  .telemetry{
    position:absolute; left:calc(12px + env(safe-area-inset-left));
    bottom:calc(16px + env(safe-area-inset-bottom));
    display:flex; flex-direction:column; gap:6px; width:150px;
  }

  /* Horn and speak: left side, above the telemetry, so the thumb that is not
     driving can reach them. Held down, not toggled. */
  .noise{
    position:absolute; left:calc(12px + env(safe-area-inset-left));
    bottom:calc(150px + env(safe-area-inset-bottom));
    display:flex; flex-direction:column; gap:10px;
  }
  .hold{
    width:74px; height:74px; border-radius:50%; touch-action:none;
    border:1px solid rgba(255,255,255,.2); background:rgba(13,17,23,.6);
    -webkit-backdrop-filter:blur(8px); backdrop-filter:blur(8px);
    color:var(--ink); font-size:11px; font-weight:700; letter-spacing:.04em;
    cursor:pointer; display:flex; flex-direction:column; align-items:center;
    justify-content:center; gap:3px; line-height:1; user-select:none;
    -webkit-user-select:none;
  }
  .hold .glyph{font-size:22px;line-height:1}
  .hold.live{background:rgba(63,185,80,.85);border-color:#fff;color:#08130c;
              box-shadow:0 0 18px rgba(63,185,80,.6)}
  .hold.busy{opacity:.6}
  .hold[disabled]{opacity:.35;cursor:not-allowed}
  .mtr{
    background:rgba(13,17,23,.6); border:1px solid rgba(255,255,255,.14);
    -webkit-backdrop-filter:blur(8px); backdrop-filter:blur(8px);
    border-radius:10px; padding:7px 9px;
  }
  .mtr b{font-size:10px;color:var(--muted);font-weight:600;letter-spacing:.04em}
  .mtr .val{font-size:17px;font-weight:700;font-variant-numeric:tabular-nums;
            line-height:1.1}
  .mtr .head{display:flex;align-items:baseline;justify-content:space-between;gap:6px}
  .bar{height:4px;background:rgba(0,0,0,.55);border-radius:3px;margin-top:5px;
       overflow:hidden;position:relative}
  .bar i{position:absolute;top:0;bottom:0;left:50%;width:0;background:var(--accent);transition:.06s}

  .stopbtn{
    border:0;border-radius:12px;padding:12px 22px;background:var(--danger);
    color:#fff;font-size:15px;font-weight:700;cursor:pointer;
    box-shadow:0 4px 18px rgba(248,81,73,.35);
  }

  /* ---- settings sheet, over the feed ---- */
  .sheet{
    position:absolute;inset:0;z-index:3;display:none;overflow:auto;
    background:rgba(13,17,23,.94);
    -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px);
    padding:calc(16px + env(safe-area-inset-top)) 16px calc(24px + env(safe-area-inset-bottom));
  }
  .sheet.open{display:block}
  .sheet .inner{max-width:460px;margin:0 auto}
  .sheet .bar-top{display:flex;align-items:center;gap:10px;margin-bottom:14px;
    position:sticky;top:0;z-index:1;padding:6px 0;background:rgba(13,17,23,.94)}
  .sheet .bar-top h2{font-size:16px;margin:0;flex:1}
  label{display:block;font-size:12px;color:var(--muted);margin-bottom:4px}
  input,select{width:100%;background:#0d1117;border:1px solid var(--line);color:var(--ink);
        border-radius:8px;padding:9px 10px;font-size:14px}
  .row{display:flex;gap:10px}
  .row>div{flex:1}
  .hint{color:var(--muted);font-size:12px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
  .chk{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--ink)}
  .chk input{width:auto}
</style>
</head>
<body>

<img id="feed" alt="camera feed">
<canvas id="depth"></canvas>
<div id="feedmsg" class="show"><div class="spin"></div><span id="feedtext">connecting to camera&hellip;</span></div>
<audio id="audio"></audio>

<div class="hud">

  <div class="top">
    <span class="dot" id="dot"></span>
    <h1>Leonida</h1>
    <button class="btn" id="mute" title="toggle audio (M)">&#128263; Audio off</button>
    <button class="btn" id="gear" title="settings">&#9881;</button>
  </div>

  <div class="noise">
    <button class="hold" id="hornbtn" title="hold to sound the horn (H)">
      <span class="glyph">&#128266;</span><span>HORN</span></button>
    <button class="hold" id="speakbtn" title="hold to talk through the car (V)">
      <span class="glyph">&#127908;</span><span>SPEAK</span></button>
  </div>

  <div class="telemetry">
    <div class="mtr"><div class="head"><b>M1 FRONT</b><span class="val" id="v1">0</span></div><div class="bar"><i id="b1"></i></div></div>
    <div class="mtr"><div class="head"><b>M2 REAR</b><span class="val" id="v2">0</span></div><div class="bar"><i id="b2"></i></div></div>
    <div class="mtr"><div class="head"><b>M3 STEER</b><span class="val" id="v3">0</span></div><div class="bar"><i id="b3"></i></div></div>
  </div>

  <div class="padwrap">
    <button class="stopbtn" id="stop">&#9632; STOP</button>
    <div id="pad"><div id="knob"></div></div>
  </div>

  <div class="sheet" id="sheet">
    <div class="inner">
      <div class="bar-top">
        <h2>&#9881; Settings</h2>
        <button class="btn" id="close">Done</button>
      </div>

      <div class="grid2" style="margin-top:0">
        <label class="chk"><input type="checkbox" id="camon" checked> Camera feed on</label>
        <label class="chk"><input type="checkbox" id="fit"> Fit whole frame (no crop)</label>
      </div>

      <div class="grid2">
        <label class="chk"><input type="checkbox" id="depthon"> Depth overlay</label>
        <div>
          <label>Depth opacity (%)</label>
          <input id="depthop" type="number" min="10" max="100" value="55">
        </div>
      </div>
      <p class="hint" id="depthnote" style="text-align:left;margin:6px 0 0">
        Runs a depth model in this browser, not on the car &ndash; the Pi keeps
        its cores for driving. Needs WebGPU to be usable.
      </p>

      <div class="row" style="margin-top:14px">
        <div>
          <label>Drive speed, min &ndash; max (%)</label>
          <div class="row">
            <div><input id="minspd" type="number" min="0" max="99" value="0"></div>
            <div><input id="maxspd" type="number" min="10" max="100" value="100"></div>
          </div>
        </div>
      </div>
      <div class="row" style="margin-top:10px">
        <div>
          <label>Steer power, min &ndash; max (%)</label>
          <div class="row">
            <div><input id="minsteer" type="number" min="0" max="99" value="0"></div>
            <div><input id="maxsteer" type="number" min="10" max="100" value="100"></div>
          </div>
        </div>
      </div>
      <div class="row" style="margin-top:10px">
        <div>
          <label>Deadzone (%)</label>
          <input id="deadzone" type="number" min="0" max="40" value="8">
        </div>
      </div>
      <div class="grid2">
        <label class="chk"><input type="checkbox" id="inv1"> Invert Motor 1 (front)</label>
        <label class="chk"><input type="checkbox" id="inv2"> Invert Motor 2 (rear)</label>
        <label class="chk"><input type="checkbox" id="inv3"> Invert steering</label>
      </div>
      <p class="hint" style="text-align:left;margin:10px 0 0">
        Motors 1 and 2 are the front and rear drive motors: they always get the
        same speed and direction, set by how far the stick is from the centre in
        any direction &ndash; full left is still full power. Forward vs reverse
        comes from which half of the pad you are in. Motor 3 steers left/right,
        scaled by how far the stick is from the vertical centre line.
        If a drive motor is wired backwards and fights the other one,
        tick its invert box; if the car steers the wrong way, invert steering.
      </p>
      <p class="hint" style="text-align:left;margin:8px 0 0">
        <b>Min</b> is the lowest power that actually makes the motor turn. Any
        stick input past the deadzone starts at min and scales up to max, so
        there is no dead band where the motor just buzzes. Raise it until the
        smallest nudge moves the car.
      </p>
      <p class="hint" style="text-align:left;margin:8px 0 0">
        Any number of people can watch at once. The camera and mic are only
        powered while someone is looking, so turning the camera off, or leaving
        audio muted, lets them rest.
      </p>
      <p class="hint" style="text-align:left;margin:8px 0 0">
        The car stops itself if it stops hearing from this page for about half a
        second &ndash; if you drive out of Wi-Fi range it will not keep going.
        Keys: <b>W A S D</b> / arrows to drive, <b>M</b> mutes, <b>space</b> stops.
      </p>
    </div>
  </div>

</div>

<!-- The depth worker. Kept as text and started from a blob so that the whole
     program remains one file you can copy to a Pi and run. Everything in here
     runs on its own thread: the page's main thread must stay free, because
     that is where the joystick lives and where the keepalive that stops the
     car from cutting out is sent. -->
<script id="depthworker" type="text/plain">
let ort = null, session = null, W = 0, H = 0, cv = null, cx = null, busy = false;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

// Depth Anything is a vision transformer with a patch size of 14, so both
// sides of the input have to be a multiple of that. We fit the camera's own
// aspect rather than squashing it into a square -- costs nothing measurable
// and stops the depth being wrong at the edges.
const SHORT = 252;
function planSize(w, h){
  const round14 = v => Math.max(14, Math.round(v / 14) * 14);
  return h <= w ? [round14(w / h * SHORT), SHORT] : [SHORT, round14(h / w * SHORT)];
}

self.onmessage = async e => {
  const m = e.data;
  if (m.type === 'init') return init(m);
  if (m.type === 'frame') return frame(m.buf);
};

async function init(m){
  try {
    if (!self.navigator || !navigator.gpu){
      return post({type:'error', fatal:true,
                   msg:'this browser has no WebGPU, so depth would run at about one frame a second'});
    }
    ort = await import(m.base + 'ort.webgpu.bundle.min.mjs');
    ort.env.wasm.wasmPaths = m.base;
    ort.env.wasm.numThreads = 1;
    const t0 = performance.now();
    session = await ort.InferenceSession.create(m.base + m.model,
      { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
    post({type:'ready', ms: Math.round(performance.now() - t0)});
  } catch (err) {
    post({type:'error', fatal:true, msg: String(err && err.message || err).slice(0, 160)});
  }
}

async function frame(buf){
  // Always answer, even when dropping the frame. The page will not offer
  // another until it hears back, so going quiet here stops the overlay dead --
  // which is exactly what happened while the model was still loading.
  if (!session || busy) return post({type:'skip'});
  busy = true;
  const t0 = performance.now();
  try {
    const bmp = await createImageBitmap(new Blob([buf], {type:'image/jpeg'}));
    if (!W){
      [W, H] = planSize(bmp.width, bmp.height);
      cv = new OffscreenCanvas(W, H);
      cx = cv.getContext('2d', {willReadFrequently:true});
    }
    cx.drawImage(bmp, 0, 0, W, H);
    bmp.close();

    const px = cx.getImageData(0, 0, W, H).data;
    const n = W * H, f = new Float32Array(3 * n);
    for (let i = 0; i < n; i++){
      f[i]         = (px[i*4]     / 255 - MEAN[0]) / STD[0];
      f[i + n]     = (px[i*4 + 1] / 255 - MEAN[1]) / STD[1];
      f[i + 2*n]   = (px[i*4 + 2] / 255 - MEAN[2]) / STD[2];
    }
    const feeds = {};
    feeds[session.inputNames[0]] = new ort.Tensor('float32', f, [1, 3, H, W]);
    const out = await session.run(feeds);
    const r = out[session.outputNames[0]];
    const d = r.data, dh = r.dims[r.dims.length - 2], dw = r.dims[r.dims.length - 1];

    // The model gives relative inverse depth -- bigger means nearer, with no
    // fixed scale -- so each frame is stretched over its own range.
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < d.length; i++){ const v = d[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = (hi - lo) || 1;

    const img = new ImageData(dw, dh);
    for (let i = 0; i < dw * dh; i++){
      const t = (d[i] - lo) / span;          // 0 far .. 1 near
      img.data[i*4]     = 255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4*t - 3)));
      img.data[i*4 + 1] = 255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4*t - 2)));
      img.data[i*4 + 2] = 255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4*t - 1)));
      img.data[i*4 + 3] = 255;
    }
    if (!cv.depthOut || cv.depthOut.width !== dw) cv.depthOut = new OffscreenCanvas(dw, dh);
    cv.depthOut.getContext('2d').putImageData(img, 0, 0);
    const bitmap = cv.depthOut.transferToImageBitmap();
    post({type:'depth', bitmap, ms: Math.round(performance.now() - t0)}, [bitmap]);
  } catch (err) {
    post({type:'error', msg: String(err && err.message || err).slice(0, 160)});
  } finally {
    busy = false;
  }
}

function post(m, transfer){ self.postMessage(m, transfer || []); }
</script>

<!-- Microphone capture. Runs on the browser's audio thread, where it cannot be
     delayed by anything the page is doing, and hands up whole 20ms lumps so
     the main thread is woken 50 times a second rather than 375. -->
<script id="micworklet" type="text/plain">
class Mic extends AudioWorkletProcessor {
  constructor(opts){
    super();
    // The car plays 16 kHz. If the browser would not give us that rate, walk
    // through the input at a fraction of a sample per step instead -- rough,
    // but this is speech through a small speaker on a car.
    this.ratio = sampleRate / opts.processorOptions.rate;
    this.out = new Int16Array(opts.processorOptions.chunk);
    this.n = 0;
    this.pos = 0;
  }
  process(inputs){
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    let p = this.pos;
    while (p < ch.length){
      const v = ch[p | 0];
      this.out[this.n++] = Math.max(-1, Math.min(1, v)) * 32767;
      if (this.n === this.out.length){
        this.port.postMessage(this.out.slice());
        this.n = 0;
      }
      p += this.ratio;
    }
    this.pos = p - ch.length;      // carry the fraction into the next block
    return true;
  }
}
registerProcessor('mic', Mic);
</script>

<script>
(() => {
  const $ = id => document.getElementById(id);
  const pad = $('pad'), knob = $('knob'), dot = $('dot');

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const cfg = {
    max:      () => clamp(+$('maxspd').value   || 100, 10, 100),
    steer:    () => clamp(+$('maxsteer').value || 100, 10, 100),
    // min is capped below max so the range can never invert
    min:      () => clamp(+$('minspd').value   || 0, 0, cfg.max()),
    minSteer: () => clamp(+$('minsteer').value || 0, 0, cfg.steer()),
    dead:     () => clamp(+$('deadzone').value || 0, 0, 40) / 100,
    inv1:     () => $('inv1').checked ? -1 : 1,
    inv2:     () => $('inv2').checked ? -1 : 1,
    inv3:     () => $('inv3').checked ? -1 : 1,
  };

  const KEYS = ['minspd','maxspd','minsteer','maxsteer','deadzone','depthop'];
  const BOOLS = ['inv1','inv2','inv3','camon','fit','depthon'];
  try {
    const s = JSON.parse(localStorage.getItem('leonida') || '{}');
    KEYS.forEach(k => { if (s[k] != null) $(k).value = s[k]; });
    BOOLS.forEach(k => { if (s[k] != null) $(k).checked = s[k]; });
  } catch {}
  function save(){
    const s = {};
    KEYS.forEach(k => s[k] = $(k).value);
    BOOLS.forEach(k => s[k] = $(k).checked);
    localStorage.setItem('leonida', JSON.stringify(s));
  }
  [...KEYS, ...BOOLS].forEach(k => $(k).addEventListener('change', save));

  // ---- control link ------------------------------------------------------
  // One WebSocket carries every stick update: all three motors in a single
  // tiny message, over a connection that stays open. That is the whole reason
  // the car feels immediate -- no request, no response, no handshake per move.
  // If the socket will not open, or dies mid-drive, we fall back to /drive,
  // which does the same job in one HTTP request, and keep trying to get the
  // socket back.
  let ws = null, wsTries = 0, wsTimer = null, okShown = 0, lastSent = '';

  function connect(){
    wsTimer = null;
    let sock;
    try {
      const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
      sock = new WebSocket(scheme + location.host + '/ws');
    } catch { retry(); return; }
    ws = sock;
    sock.onopen    = () => { wsTries = 0; lastSent = ''; };
    sock.onmessage = () => { okShown = Date.now(); dot.classList.add('ok'); };
    sock.onerror   = () => { try { sock.close(); } catch {} };
    sock.onclose   = () => { if (ws === sock) { ws = null; retry(); } };
  }
  function retry(){
    if (wsTimer) return;
    wsTries++;
    wsTimer = setTimeout(connect, Math.min(2000, 250 * wsTries));
  }

  async function viaHttp(path){
    try {
      const r = await fetch(path, { cache:'no-store', keepalive:true });
      if (r.ok){ okShown = Date.now(); dot.classList.add('ok'); }
    } catch {}
  }

  function sendAll(p1, p2, p3){
    const msg = Math.round(p1) + ',' + Math.round(p2) + ',' + Math.round(p3);
    if (msg === lastSent) return;
    lastSent = msg;
    if (ws && ws.readyState === 1) ws.send(msg);
    else viaHttp('/drive?m1=' + Math.round(p1) + '&m2=' + Math.round(p2) +
                 '&m3=' + Math.round(p3));
  }

  // The car cuts the motors if it stops hearing from us, so keep talking even
  // when the stick has not moved -- holding a steady throttle sends nothing.
  //
  // This tick is the failsafe's heartbeat, and it is meant to stop when the
  // page does: a browser throttles timers in a backgrounded tab to a crawl, so
  // a phone going into a pocket cuts the motors even if visibilitychange never
  // fires. That is the desired behaviour, not a bug to work around -- hence no
  // document.hidden check here, which would only make the same decision worse
  // in embedded browsers that call a plainly visible page hidden.
  //
  // It runs well inside the failsafe window so that ordinary timer jitter, a
  // slow frame or a garbage collection pause, cannot stutter the motors: three
  // ticks in a row have to go missing before the car stops.
  // This tick is also the canary. If anything ever makes the main thread miss
  // its slot badly -- the depth overlay being the obvious candidate -- the
  // keepalive goes with it and the car cuts out. So watch our own punctuality,
  // and if it goes properly bad, drop the overlay rather than the throttle.
  let lastTick = Date.now(), lagStrikes = 0;
  setInterval(() => {
    const now = Date.now(), late = now - lastTick - 150;
    lastTick = now;

    if (ws && ws.readyState === 1) ws.send('p');
    else viaHttp('/ping');
    if (Date.now() - okShown > 2000) dot.classList.remove('ok');

    // Only judge our punctuality while the page is actually on screen. A
    // backgrounded tab has its timers throttled to a crawl by the browser,
    // which looks identical to a blocked main thread from in here -- and when
    // the page is hidden the car is stopped anyway, so there is nothing to
    // protect. Guessing wrong in that direction would switch the overlay off
    // every time the phone was pocketed.
    if (!depthWorker || document.hidden) { lagStrikes = 0; return; }
    if (late > 400){
      if (++lagStrikes >= 3){
        $('depthon').checked = false; save(); depthStart();
        depthSay('depth turned itself off: it was delaying the controls, and ' +
                 'driving comes first');
      }
    } else if (late < 150) {
      lagStrikes = 0;
    }
  }, 150);

  connect();

  let X = 0, Y = 0;
  // Axes are independent here, so the deadzone is per-axis (not on the vector
  // magnitude) -- otherwise a small steer input would also kill the throttle.
  function dz(v){
    const d = cfg.dead();
    if (Math.abs(v) < d) return 0;
    return Math.sign(v) * (Math.abs(v) - d) / (1 - d);
  }
  // Motors stall below some power, so a live input never maps to less than
  // `min` -- 0 stays a true stop, anything else lands in [min, max].
  function power(v, min, max){
    if (v === 0) return 0;
    return Math.sign(v) * (min + Math.abs(v) * (max - min));
  }
  function apply(){
    // Drive is polar: its magnitude is the stick's distance from the centre,
    // regardless of direction -- so a stick pushed fully left still means full
    // power. Only the sign comes from the Y axis (on the centre line, forward).
    const r = Math.min(1, Math.hypot(X, Y));
    const throttle = dz(r) * (Y < 0 ? -1 : 1);
    // Steering is the horizontal distance from the pad's vertical centre line.
    const steer = dz(X);

    // Motors 1 and 2 are the front and rear drive motors: identical speed and
    // direction, straight off the throttle.
    const drive = power(throttle, cfg.min(), cfg.max());
    const p1 = drive * cfg.inv1();
    const p2 = drive * cfg.inv2();
    // Motor 3 is the steering motor.
    const p3 = power(steer, cfg.minSteer(), cfg.steer()) * cfg.inv3();

    ui(1,p1); ui(2,p2); ui(3,p3);
    sendAll(p1, p2, p3);
  }
  function ui(n, p){
    $('v'+n).textContent = Math.round(p);
    const bar = $('b'+n), w = Math.min(50, Math.abs(p)/2);
    bar.style.width = w + '%';
    bar.style.left = p >= 0 ? '50%' : (50 - w) + '%';
    bar.style.background = p >= 0 ? 'var(--accent)' : 'var(--accent2)';
  }
  function drawKnob(){
    const r = pad.clientWidth * 0.33;
    knob.style.transform = `translate(${X*r}px, ${-Y*r}px)`;
  }

  let active = false, pid = null;
  function setFromEvent(e){
    const rect = pad.getBoundingClientRect();
    const cx = rect.left + rect.width/2, cy = rect.top + rect.height/2;
    let nx = (e.clientX - cx) / (rect.width/2);
    let ny = (e.clientY - cy) / (rect.height/2);
    const mag = Math.hypot(nx, ny);
    if (mag > 1){ nx /= mag; ny /= mag; }
    X = nx; Y = -ny; drawKnob(); apply();
  }
  pad.addEventListener('pointerdown', e => {
    active = true; pid = e.pointerId; pad.setPointerCapture(pid); setFromEvent(e);
  });
  pad.addEventListener('pointermove', e => { if (active && e.pointerId === pid) setFromEvent(e); });
  function release(e){
    if (!active || (e && e.pointerId !== pid)) return;
    active = false; pid = null; X = 0; Y = 0; drawKnob(); apply();
  }
  pad.addEventListener('pointerup', release);
  pad.addEventListener('pointercancel', release);

  const keys = new Set();
  const KMAP = {'w':'f','arrowup':'f','s':'b','arrowdown':'b',
                'a':'l','arrowleft':'l','d':'r','arrowright':'r'};
  function fromKeys(){
    X = (keys.has('r')?1:0) - (keys.has('l')?1:0);
    Y = (keys.has('f')?1:0) - (keys.has('b')?1:0);
    drawKnob(); apply();
  }
  const typing = e => e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT';
  addEventListener('keydown', e => {
    if (typing(e)) return;
    const k = KMAP[e.key.toLowerCase()]; if (!k) return; e.preventDefault();
    if (!keys.has(k)){ keys.add(k); fromKeys(); }
  });
  addEventListener('keyup', e => {
    const k = KMAP[e.key.toLowerCase()]; if (!k) return; keys.delete(k); fromKeys();
  });

  function stopAll(){
    keys.clear(); X = 0; Y = 0; drawKnob();
    lastSent = '';                 // force the stop out even if we just sent 0
    sendAll(0, 0, 0);
    ui(1,0); ui(2,0); ui(3,0);
    // The car silences the horn whenever it stops the wheels, so the button
    // has to stop claiming otherwise.
    horn(false);
  }
  $('stop').addEventListener('click', stopAll);
  addEventListener('blur', () => { if (active || keys.size) stopAll(); });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden){ stopAll(); return; }
    // Coming back: get the socket up now rather than waiting out the backoff,
    // so unlocking the phone and driving again feels immediate.
    if (!ws){ clearTimeout(wsTimer); wsTimer = null; wsTries = 0; connect(); }
  });

  drawKnob();

  // ---- camera feed -------------------------------------------------------
  // Frames arrive over a WebSocket, one whole JPEG per binary message, and go
  // into the <img> as blob URLs.
  //
  // Both of the obvious ways to show a live camera fail on Safari, which means
  // they fail on every iPhone -- the exact machine this page is meant for.
  // An <img> pointed at a multipart/x-mixed-replace stream is not rendered at
  // all; reading that same stream with fetch() needs streaming response bodies,
  // which Safari did not have until 14.1 and still treats specially for
  // multipart. A binary WebSocket has worked since Safari 6 and iOS 6, needs no
  // parsing here, and pushes frames rather than waiting to be asked.
  //
  // If a WebSocket cannot be had at all, we fall back to reloading a plain
  // JPEG from /snapshot.jpg, which is just an <img> loading an ordinary image
  // and works in anything that can show a picture.
  const feed = $('feed'), feedmsg = $('feedmsg'), feedtext = $('feedtext');
  let camWs = null, camTimer = null, camTries = 0, camPolling = false;
  let camBadFrames = 0;
  const camUrls = [];

  function say(text, spinning){
    feedtext.textContent = text;
    feedmsg.querySelector('.spin').style.visibility = spinning ? '' : 'hidden';
    feedmsg.classList.add('show');
  }

  function camLive(){
    if (feed.classList.contains('dead')){      // first frame through
      camTries = 0; feed.classList.remove('dead'); feedmsg.classList.remove('show');
    }
  }

  // Handed straight to the <img>, not queued behind requestAnimationFrame:
  // rAF does not fire in a backgrounded or throttled tab, and a feed that
  // freezes the moment the phone thinks you are not looking is no feed at all.
  // A frame the browser has not finished decoding is simply replaced by the
  // next one, which is what we want anyway -- only the newest frame matters.
  function showFrame(bytes){
    const url = URL.createObjectURL(new Blob([bytes], {type:'image/jpeg'}));
    feed.src = url;
    // Keep the newest few alive rather than revoking the frame we just
    // replaced: the <img> may still be decoding it, and pulling a blob URL out
    // from under Safari mid-decode blanks the picture.
    camUrls.push(url);
    while (camUrls.length > 3) URL.revokeObjectURL(camUrls.shift());
  }

  // An <img> that cannot decode what it was given shows a broken-image icon
  // and says nothing about why, which is a miserable thing to debug from the
  // driving seat. If frames are plainly arriving and none of them decode, say
  // exactly that instead of leaving a question mark sitting there.
  function camWatchDecode(){
    feed.onload = () => {
      if (camPolling) return;
      camBadFrames = 0;
      camLive();
    };
    feed.onerror = () => {
      if (camPolling) return;
      if (++camBadFrames === 8){
        say('frames are arriving, but this browser cannot decode them ' +
            '— see "camera" in the README', false);
      }
    };
  }

  function camStop(){
    clearTimeout(camTimer); camTimer = null;
    camPolling = false;
    camBadFrames = 0;
    feed.onload = feed.onerror = null;
    if (camWs){ const s = camWs; camWs = null; try { s.close(); } catch {} }
    feed.removeAttribute('src');
    while (camUrls.length) URL.revokeObjectURL(camUrls.pop());
    feed.classList.add('dead');
  }

  function camRetry(why){
    if (!$('camon').checked) return;
    camTries++;
    say('no camera — ' + (why || 'retrying') + ' — retrying…', true);
    camTimer = setTimeout(camStart, Math.min(8000, 1000 * camTries));
  }

  function camStart(){
    camStop();
    if (!$('camon').checked){ say('camera off', false); return; }
    say(camTries ? 'reconnecting to camera…' : 'connecting to camera…', true);

    if (!window.WebSocket){ camPollStart(); return; }
    let sock;
    try {
      const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
      sock = new WebSocket(scheme + location.host + '/ws/video');
    } catch { camPollStart(); return; }
    sock.binaryType = 'arraybuffer';
    camWs = sock;
    camWatchDecode();

    sock.onmessage = ev => {
      if (typeof ev.data === 'string'){        // the car explaining itself
        let why = ev.data;
        try { const j = JSON.parse(ev.data); if (j && j.error) why = j.error; } catch {}
        camWs = null; try { sock.close(); } catch {}
        camRetry(why);
        return;
      }
      showFrame(ev.data);
      // showFrame copied the bytes into a Blob, so the buffer is still ours to
      // hand over -- transferred, not copied, and dropped if the model is busy.
      depthOffer(ev.data);
    };
    sock.onerror = () => { try { sock.close(); } catch {} };
    sock.onclose = () => {
      if (camWs !== sock) return;              // we tore it down on purpose
      camWs = null;
      // Never got a single frame: the socket itself may be the problem, so
      // drop to the one method that cannot be blocked.
      if (feed.classList.contains('dead')) camPollStart();
      else camRetry('the camera stopped sending');
    };
  }

  function camPollStart(){
    camPolling = true;
    say('connecting to camera… (snapshot mode)', true);
    feed.onload = () => {
      if (!camPolling) return;
      camLive();
      camTimer = setTimeout(camPollNext, 60);
    };
    feed.onerror = () => {
      if (!camPolling) return;
      camPolling = false;
      feed.onload = feed.onerror = null;
      camRetry('snapshot failed');
    };
    camPollNext();
  }
  function camPollNext(){
    if (!camPolling || !$('camon').checked) return;
    feed.src = '/snapshot.jpg?t=' + Date.now();
  }
  // ---- horn and speak ----------------------------------------------------
  // Both are held, not toggled, and both are released by anything that takes
  // the page away -- the car should never be left making a noise nobody meant.
  const SPEAK_RATE = 16000, SPEAK_CHUNK = 320;   // 20ms, matching the car
  const hornBtn = $('hornbtn'), speakBtn = $('speakbtn');
  let hornOn = false;

  function horn(on){
    if (on === hornOn) return;
    hornOn = on;
    hornBtn.classList.toggle('live', on);
    if (ws && ws.readyState === 1) ws.send(on ? 'h1' : 'h0');
    else viaHttp('/horn?on=' + (on ? 1 : 0));
  }

  // ---- speak: this browser's microphone, out of the car's speaker ----
  let micWs = null, micCtx = null, micNode = null, micStream = null;
  let speaking = false, micBusy = false, wantSpeak = false;

  function speakLabel(text){ speakBtn.title = text; }

  async function speakStart(){
    wantSpeak = true;
    if (speaking || micBusy) return;
    // getUserMedia only exists in a secure context. Over plain http the whole
    // API is simply absent, which is a confusing way to find out, so say it.
    if (!navigator.mediaDevices || !window.isSecureContext){
      speakBtn.disabled = true;
      speakLabel('the microphone needs https -- run: python3 leonida.py --make-cert');
      return;
    }
    micBusy = true;
    speakBtn.classList.add('busy');
    try {
      micStream = await navigator.mediaDevices.getUserMedia({audio:{
        echoCancellation:true, noiseSuppression:true,
        autoGainControl:true, channelCount:1}});

      micCtx = new (window.AudioContext || window.webkitAudioContext)(
        {sampleRate: SPEAK_RATE});
      const src = micCtx.createMediaStreamSource(micStream);
      const code = $('micworklet').textContent;
      const url = URL.createObjectURL(new Blob([code], {type:'text/javascript'}));
      await micCtx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);

      micWs = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://')
                            + location.host + '/ws/mic');
      micWs.binaryType = 'arraybuffer';

      micNode = new AudioWorkletNode(micCtx, 'mic', {
        processorOptions: {rate: SPEAK_RATE, chunk: SPEAK_CHUNK}});
      micNode.port.onmessage = ev => {
        if (micWs && micWs.readyState === 1) micWs.send(ev.data.buffer);
      };
      src.connect(micNode);
      // Not connected to the destination on purpose: routing the microphone to
      // this phone's own speaker is how you start a feedback loop.

      // Asking for the microphone takes a moment, and the first time it waits
      // on a permission dialog. If the button was let go in the meantime, this
      // must not come up live and stay that way.
      if (!wantSpeak){ speakStop(); return; }

      speaking = true;
      speakBtn.classList.add('live');
      speakLabel('talking through the car');
      // The car's own microphone playing here while the car's speaker is
      // talking is the other way to start one.
      if (listening){ audio.pause(); }
    } catch (err) {
      speakLabel(err && err.name === 'NotAllowedError'
                 ? 'microphone permission refused'
                 : 'microphone unavailable: ' + (err && err.message || err));
      speakStop();
    } finally {
      micBusy = false;
      speakBtn.classList.remove('busy');
    }
  }

  function speakStop(){
    wantSpeak = false;
    speaking = false;
    speakBtn.classList.remove('live');
    if (micNode){ try { micNode.port.onmessage = null; micNode.disconnect(); } catch {} micNode = null; }
    if (micStream){ micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (micCtx){ const c = micCtx; micCtx = null; c.close().catch(() => {}); }
    if (micWs){ const s = micWs; micWs = null; try { s.close(); } catch {} }
    if (listening) audio.play().catch(() => {});      // give the car's ears back
  }

  // ---- holding the buttons ----
  function holdable(btn, down, up){
    let held = false;
    btn.addEventListener('pointerdown', e => {
      if (btn.disabled) return;
      e.preventDefault();                 // no focus, so space still means stop
      held = true;
      try { btn.setPointerCapture(e.pointerId); } catch {}
      down();
    });
    const release = () => { if (held){ held = false; up(); } };
    btn.addEventListener('pointerup', release);
    btn.addEventListener('pointercancel', release);
    // Also on the window: if capture did not take, the release can land
    // somewhere else entirely, and a horn that never hears about it is stuck.
    addEventListener('pointerup', release);
    addEventListener('pointercancel', release);
    btn.addEventListener('contextmenu', e => e.preventDefault());
    return release;
  }
  const hornRelease = holdable(hornBtn, () => horn(true), () => horn(false));
  const speakRelease = holdable(speakBtn, speakStart, speakStop);

  // Same two, from the keyboard. Held down, and repeat events ignored.
  let hKey = false, vKey = false;
  addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    const k = e.key.toLowerCase();
    if (k === 'h' && !hKey){ hKey = true; e.preventDefault(); horn(true); }
    else if (k === 'v' && !vKey){ vKey = true; e.preventDefault(); speakStart(); }
  });
  addEventListener('keyup', e => {
    const k = e.key.toLowerCase();
    if (k === 'h'){ hKey = false; horn(false); }
    else if (k === 'v'){ vKey = false; speakStop(); }
  });

  function quiet(){ horn(false); speakStop(); }
  addEventListener('blur', quiet);
  addEventListener('pagehide', quiet);
  document.addEventListener('visibilitychange', () => { if (document.hidden) quiet(); });

  if (!window.isSecureContext){
    speakBtn.disabled = true;
    speakLabel('the microphone needs https -- run: python3 leonida.py --make-cert');
  }

  // ---- depth overlay -----------------------------------------------------
  // The model runs in the worker above, in this browser, on frames that have
  // already arrived for display. The car does no part of it and cannot be
  // slowed down by it. All this thread does is hand over a buffer and draw the
  // picture that comes back.
  const depthCv = $('depth'), depthCx = depthCv.getContext('2d');
  let depthWorker = null, depthInFlight = false, depthMs = 0;

  function depthSay(text){ $('depthnote').textContent = text; }

  function depthStop(){
    if (depthWorker){ depthWorker.terminate(); depthWorker = null; }
    depthInFlight = false;
    depthCv.classList.remove('on');
  }

  function depthStart(){
    depthStop();
    if (!$('depthon').checked){
      depthSay('Runs a depth model in this browser, not on the car — the Pi ' +
               'keeps its cores for driving. Needs WebGPU to be usable.');
      return;
    }
    if (!window.Worker || !window.OffscreenCanvas){
      $('depthon').checked = false;
      depthSay('This browser is missing the pieces needed to run the model.');
      return;
    }
    depthSay('loading the depth model…');
    const src = $('depthworker').textContent;
    const url = URL.createObjectURL(new Blob([src], {type:'text/javascript'}));
    try {
      depthWorker = new Worker(url, {type:'module'});
    } catch (e) {
      $('depthon').checked = false;
      depthSay('could not start the depth worker: ' + e.message);
      return;
    }
    URL.revokeObjectURL(url);

    depthWorker.onmessage = ev => {
      const m = ev.data;
      if (m.type === 'skip'){
        depthInFlight = false;
      } else if (m.type === 'ready'){
        depthSay('depth model ready (' + m.ms + ' ms to load) — running here, ' +
                 'not on the car');
        depthCv.classList.add('on');
      } else if (m.type === 'depth'){
        depthInFlight = false;
        depthMs = m.ms;
        if (depthCv.width !== m.bitmap.width){
          depthCv.width = m.bitmap.width; depthCv.height = m.bitmap.height;
        }
        depthCx.drawImage(m.bitmap, 0, 0);
        m.bitmap.close();
        depthSay('depth: ' + m.ms + ' ms per frame (' +
                 (1000 / Math.max(1, m.ms)).toFixed(0) + ' fps) in this browser');
      } else if (m.type === 'error'){
        depthInFlight = false;
        depthSay('depth off — ' + m.msg);
        if (m.fatal){ $('depthon').checked = false; save(); depthStop(); }
      }
    };
    depthWorker.onerror = e => {
      depthSay('depth worker failed: ' + (e.message || 'unknown'));
      $('depthon').checked = false; depthStop();
    };
    depthWorker.postMessage({type:'init', base: location.origin + '/assets/',
                             model: 'DEPTH_MODEL_NAME'});
  }

  // Given a frame that has just arrived, offer it to the model. Never queues:
  // if the worker is still busy this frame is simply skipped, so the overlay
  // runs as fast as the device allows and no faster.
  function depthOffer(buf){
    if (!depthWorker || depthInFlight) return;
    depthInFlight = true;
    depthWorker.postMessage({type:'frame', buf}, [buf]);
  }

  function depthOpacity(){
    depthCv.style.opacity = clamp(+$('depthop').value || 55, 10, 100) / 100;
  }
  $('depthop').addEventListener('input', depthOpacity);
  depthOpacity();
  $('depthon').addEventListener('change', depthStart);

  $('camon').addEventListener('change', () => { camTries = 0; camStart(); });
  function applyFit(){
    const how = $('fit').checked ? 'contain' : 'cover';
    feed.style.objectFit = how;
    depthCv.style.objectFit = how;      // must crop exactly like the feed does
  }
  $('fit').addEventListener('change', applyFit);
  applyFit();

  // ---- audio -------------------------------------------------------------
  // Muting tears the stream down rather than silencing it, so the mic is only
  // held while someone is listening. Unmuting is always a tap, which is also
  // what browsers require before audio may play.
  const audio = $('audio'), muteBtn = $('mute');
  let listening = false;

  function setMute(on){
    listening = !on;
    if (listening){
      audio.src = '/stream/audio?t=' + Date.now();
      audio.play().catch(() => setMute(true));   // browser refused -> stay muted
      muteBtn.classList.add('on');
      muteBtn.innerHTML = '🔊 Audio on';
    } else {
      audio.pause();
      audio.removeAttribute('src');
      audio.load();                              // drops the connection
      muteBtn.classList.remove('on');
      muteBtn.innerHTML = '🔇 Audio off';
    }
  }
  muteBtn.addEventListener('click', () => setMute(listening));
  audio.addEventListener('error', () => { if (listening) setMute(true); });
  setMute(true);

  // ---- settings sheet ----------------------------------------------------
  const sheet = $('sheet');
  $('gear').addEventListener('click', () => {
    sheet.classList.add('open'); sheet.scrollTop = 0;
  });
  $('close').addEventListener('click', () => sheet.classList.remove('open'));

  addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    const k = e.key.toLowerCase();
    if (k === 'm'){ e.preventDefault(); setMute(listening); }
    else if (k === ' '){ e.preventDefault(); stopAll(); }
    else if (k === 'escape') sheet.classList.remove('open');
  });

  // Let go of the car and its camera cleanly when the tab closes.
  addEventListener('pagehide', () => {
    stopAll();
    camStop(); setMute(true); depthStop();
    if (ws){ const s = ws; ws = null; try { s.close(); } catch {} }
  });

  camStart();
  depthStart();
})();
</script>
</body>
</html>
"""


# The page names the model file; keep that in one place.
PAGE = PAGE.replace("DEPTH_MODEL_NAME", DEPTH_MODEL)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True     # a 15-byte drive command should not wait

    def log_message(self, *args):
        pass                           # keep the console to motor state only

    # ---- plumbing ----

    def _send(self, status, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload), "application/json")

    def _send_asset(self, name):
        """Serve one downloaded runtime or model file.

        These are tens of megabytes and never change, so unlike everything
        else here they are cached hard -- a phone that re-fetched the model on
        every page load would be unusable.
        """
        # No directory traversal: a bare filename from our own manifest only.
        if name != os.path.basename(name) or name.startswith("."):
            self._json(404, {"error": "not found"})
            return
        if name not in [n for _u, n in ASSET_SOURCES]:
            self._json(404, {"error": f"unknown asset: {name}"})
            return

        path = os.path.join(ASSET_DIR, name)
        try:
            size = os.path.getsize(path)
            body = open(path, "rb")
        except OSError:
            self._json(503, {"error": "depth assets are not downloaded -- run "
                                      "python3 leonida.py --fetch-model"})
            return

        ext = os.path.splitext(name)[1]
        with body:
            self.send_response(200)
            self.send_header("Content-Type",
                             ASSET_TYPES.get(ext, "application/octet-stream"))
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            while True:
                chunk = body.read(262144)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _send_cert(self):
        """Hand out the certificate so a phone can be told to trust it.

        The certificate only -- it is public by nature, it is sent to every
        client during every handshake anyway. The private key next to it is
        never served by any route here, and that is deliberate.
        """
        try:
            with open(CFG.cert, "rb") as f:
                body = f.read()
        except OSError:
            self._json(404, {"error": "no certificate -- run "
                                      "python3 leonida.py --make-cert"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-pem-file")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition",
                         'attachment; filename="leonida-cert.pem"')
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY:
            return None
        return self.rfile.read(length)

    # ---- routing ----

    def _plaintext_on_a_tls_port(self):
        """True if this request arrived unencrypted on an HTTPS server."""
        return (self.server.tls is not None
                and not isinstance(self.connection, ssl.SSLSocket))

    def _redirect_to_https(self):
        host = self.headers.get("Host") or f"{local_ip()}:{CFG.port}"
        body = b"This car speaks HTTPS.\n"
        self.send_response(301)
        self.send_header("Location", f"https://{host}{self.path}")
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self._plaintext_on_a_tls_port():
            self._redirect_to_https()
            return

        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/ws":
            self._websocket()
        elif path == "/ws/video":
            self._video_socket()
        elif path == "/ws/mic":
            self._mic_socket()
        elif path in ("/snapshot.jpg", "/snapshot"):
            self._snapshot()
        elif path.startswith("/assets/"):
            self._send_asset(path[len("/assets/"):])
        elif path == "/leonida-cert.pem":
            self._send_cert()
        elif path == "/drive":
            self._drive(parsed.query)
        elif path == "/ping":
            touch()
            self._json(200, {"ok": True})
        elif path == "/horn":
            self._horn(parsed.query)
        elif path == "/health":
            self._health()
        elif path in STREAMS:
            self._relay(STREAMS[path])
        elif path in ("/stream/video", "/stream/audio"):
            # Switched off at the command line. Say so plainly -- the page shows
            # this text, and "unknown endpoint" would send someone hunting for a
            # broken camera that was never turned on.
            kind, flag = (("camera", "--no-camera") if path.endswith("video")
                          else ("microphone", "--no-audio"))
            self._json(503, {"error": f"the {kind} is switched off on the car "
                                      f"({flag})"})
        elif path in ("/motor1", "/motor2", "/motor3"):
            self._motor(int(path[-1]), parsed.query)
        else:
            self._json(404, {"error": f"unknown endpoint: {parsed.path}"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path in ("/motor1", "/motor2", "/motor3"):
            self._motor(int(path[-1]), parsed.query)
        elif path == "/drive":
            self._drive(parsed.query)
        else:
            self._read_body()          # drain so the connection stays usable
            self._json(404, {"error": f"unknown endpoint: {parsed.path}"})

    do_PUT = do_POST

    # ---- control ----

    def _drive(self, query):
        """All three motors in one request -- the HTTP fallback for the page."""
        params = parse_qs(query)
        powers = []
        for name in ("m1", "m2", "m3"):
            values = params.get(name)
            percent, error = coerce_power(values[0] if values else None)
            if error is not None:
                self._json(400, {"error": f"{name}: {error}"})
                return
            powers.append(percent)
        set_all(*powers)
        self._json(200, {"m1": powers[0], "m2": powers[1], "m3": powers[2]})

    def _horn(self, query):
        """The horn, for the HTTP fallback path. Held, not toggled."""
        values = parse_qs(query).get("on")
        raw = (values[0] if values else "").strip().lower()
        if raw not in ("0", "1", "true", "false", "on", "off"):
            self._json(400, {"error": "on must be 0 or 1"})
            return
        on = raw in ("1", "true", "on")
        SPEAKER.set_horn(on)
        touch()
        self._json(200, {"horn": on})

    def _motor(self, index, query):
        """One motor, the way the old motor API did it."""
        values = parse_qs(query).get("power")
        raw = values[0] if values else None

        if raw is None:
            body = self._read_body()
            if body:
                try:
                    parsed = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    self._json(400, {"error": "body is not valid JSON"})
                    return
                if not isinstance(parsed, dict) or "power" not in parsed:
                    self._json(400, {"error": "missing required parameter: power"})
                    return
                raw = parsed["power"]

        percent, error = coerce_power(raw)
        if error is not None:
            self._json(400, {"error": error})
            return
        set_one(index, percent)
        self._json(200, {"motor": index, "power": percent})

    def _health(self):
        with lock:
            state = dict(_state)
        self._json(200, {
            "mock": MOCK,
            "motors": {f"motor{i}": p for i, p in state.items()},
            "controllers": CONTROLLERS.count(),
            "viewers": {name.rsplit("/", 1)[-1]: stream.viewers()
                        for name, stream in STREAMS.items()},
            "since_last_command": round(time.monotonic() - _last_command, 3),
            "deadman": DEADMAN,
            "depth_assets": depth_ready(),
            "speaker": SPEAKER.status(),
        })

    # ---- websocket ----

    def _ws_upgrade(self):
        """Answer the handshake. False if this was not a websocket request."""
        key = self.headers.get("Sec-WebSocket-Key")
        upgrade = (self.headers.get("Upgrade") or "").lower()
        if not key or "websocket" not in upgrade:
            self._json(400, {"error": "not a websocket handshake"})
            return False

        self.close_connection = True
        self.wfile.write(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: " + ws_accept(key).encode("ascii") +
            b"\r\n\r\n")
        return True

    def _websocket(self):
        if not self._ws_upgrade():
            return

        peer = self.client_address[0]
        CONTROLLERS.join()
        print(f"  + control socket from {peer}")

        # Two different kinds of "still there", and they must not be confused:
        #
        #   the socket  is kept alive by protocol ping/pong below. A browser
        #               answers a ping in its network stack, without running a
        #               line of page script, so a backgrounded tab holds its
        #               connection instead of being dropped and reconnected
        #               every few seconds.
        #
        #   the driver  is proved only by an application message ('p' or a
        #               drive command), which the page can only send while its
        #               script is actually running. That, and nothing else,
        #               feeds the motor watchdog -- so a phone that has gone
        #               into a pocket stops the car even though its socket is
        #               still perfectly healthy.
        conn = WSConn(self.connection)
        last_ack = 0.0
        quiet = 0
        try:
            while True:
                try:
                    opcode, payload = conn.read(WS_PING)
                except TIMED_OUT:
                    quiet += 1
                    if quiet > WS_QUIET:
                        break                          # nothing home at all
                    conn.write(b"", 0x9)               # ping; stack auto-pongs
                    continue
                quiet = 0

                if opcode == 0x8:                      # close
                    break
                if opcode == 0x9:                      # ping
                    conn.write(payload, 0xA)
                    continue
                if opcode != 0x1:                      # pong, or fragments
                    continue

                message = payload.decode("utf-8", "replace")
                if message == "p":
                    touch()
                elif message == "h1":
                    SPEAKER.set_horn(True)
                    touch()
                elif message == "h0":
                    SPEAKER.set_horn(False)
                    touch()
                elif not drive_from_text(message):
                    continue

                # Acking lets the page show a dot that means "the hardware
                # heard me", not merely "a socket is open". Throttled, because
                # the stick can outrun any sensible refresh rate.
                now = time.monotonic()
                if now - last_ack > 0.1:
                    last_ack = now
                    conn.write(b"k")
        except (OSError, ValueError, struct.error):
            pass
        finally:
            remaining = CONTROLLERS.leave()
            print(f"  - control socket from {peer}")
            if remaining == 0:
                stop_all("last controller disconnected")

    # ---- video over a websocket ----

    def _video_socket(self):
        """Push whole JPEGs to one browser, one per binary message.

        This is how the page actually watches the camera, because the two
        obvious alternatives both fail on Safari and so on every iPhone:
        an <img> pointed at a multipart/x-mixed-replace stream is not
        rendered at all, and reading that stream with fetch() needs streaming
        response bodies, which Safari only grew in 14.1 and still handles
        oddly for multipart. A binary WebSocket has worked since Safari 6.

        The protocol is deliberately trivial: a binary message is a frame, a
        text message is the car explaining why there are none.
        """
        if not self._ws_upgrade():
            return

        conn = WSConn(self.connection)
        stream = STREAMS.get("/stream/video")
        if stream is None:
            self._video_excuse(conn, "the camera is switched off on the car "
                                     "(--no-camera)")
            return

        att, sub, error = stream.subscribe()
        if error is not None:
            _status, body, _ctype = error
            try:
                conn.write(body)               # ffmpeg's own words, as JSON
            except OSError:
                pass
            return

        try:
            while True:
                item = sub.get(1.0)
                if item is None:
                    if sub.ended:              # the capture stopped
                        break
                    if self._video_peer_done():
                        break
                    continue                   # nothing yet, keep waiting
                if self._video_peer_done():
                    break
                conn.write(item, 0x2)          # binary: one whole JPEG
        except (OSError, ValueError, struct.error):
            pass                               # this viewer left
        finally:
            stream.unsubscribe(att, sub)

    def _video_excuse(self, conn, message):
        try:
            conn.write(json.dumps({"error": message}).encode())
        except OSError:
            pass

    def _video_peer_done(self):
        """True once the browser is finished with the feed.

        Nothing is expected from the page on this socket -- it is one way --
        so anything readable at all, a close frame or an end of stream, means
        we can let the camera go.
        """
        try:
            return bool(select.select([self.connection], [], [], 0)[0])
        except OSError:
            return True

    # ---- speech from a phone ----

    def _mic_socket(self):
        """Take raw audio from a browser and play it out of the car.

        Its own socket, not the control one: a 640-byte lump of speech must
        never sit in the queue ahead of a steering command. Binary messages are
        16 kHz mono signed 16-bit -- no codec, because decoding one would cost
        the Pi time and latency for no benefit on a local network.
        """
        if not self._ws_upgrade():
            return
        if CFG.no_speaker:
            try:
                WSConn(self.connection).write(json.dumps(
                    {"error": "the speaker is switched off on the car "
                              "(--no-speaker)"}).encode())
            except OSError:
                pass
            return

        conn = WSConn(self.connection)
        peer = self.client_address[0]
        print(f"  + speaking from {peer}")
        heard = 0
        try:
            while True:
                try:
                    opcode, payload = conn.read(WS_PING)
                except TIMED_OUT:
                    conn.write(b"", 0x9)           # still there?
                    continue
                if opcode == 0x8:                  # close
                    break
                if opcode == 0x9:
                    conn.write(payload, 0xA)
                    continue
                if opcode != 0x2:                  # only binary carries audio
                    continue
                if len(payload) % 2:
                    payload = payload[:-1]         # keep whole samples
                if payload:
                    SPEAKER.speak(payload)
                    heard += len(payload)
        except (OSError, ValueError, struct.error):
            pass
        finally:
            print(f"  - speaking from {peer} ({heard / (SPEAK_RATE * 2):.1f}s)")

    # ---- streams ----

    def _snapshot(self):
        """One JPEG, as an ordinary image response.

        The floor of the compatibility ladder: an <img> loading a normal JPEG
        works in any browser that can show a picture at all. The page polls
        this if it cannot get a websocket.
        """
        stream = STREAMS.get("/stream/video")
        if stream is None:
            self._json(503, {"error": "the camera is switched off on the car "
                                      "(--no-camera)"})
            return

        att, sub, error = stream.subscribe()
        if error is not None:
            self._send(*error)
            return
        try:
            deadline = time.monotonic() + START_WAIT
            while time.monotonic() < deadline:
                item = sub.get(1.0)
                if item is not None:
                    self._send(200, item, "image/jpeg")
                    return
                if sub.ended:
                    break
            self._json(504, {"error": "the camera sent no frame"})
        finally:
            stream.unsubscribe(att, sub)

    def _relay(self, stream):
        """Serve one browser its copy of a live feed."""
        att, sub, error = stream.subscribe()
        if error is not None:
            self._send(*error)
            return

        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", att.ctype)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()

        wrap = stream.wrap()
        try:
            while True:
                item = sub.get(1.0)
                if item is None:
                    if sub.ended:                  # the capture stopped
                        break
                    if self._viewer_gone():
                        break
                    continue                       # nothing yet, keep waiting
                if self._viewer_gone():
                    break
                data = wrap(item)
                if data:
                    self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                                   # this viewer left
        finally:
            stream.unsubscribe(att, sub)

    def _viewer_gone(self):
        """True once the browser has closed its end.

        Writing alone would not tell us for seconds -- the bytes just sit in
        the socket buffer -- and the camera keeps running until the last viewer
        is known to be gone. A closed peer shows up right away as a readable
        socket that yields EOF.
        """
        sock = self.connection
        try:
            if not select.select([sock], [], [], 0)[0]:
                return False
            if isinstance(sock, ssl.SSLSocket):
                # SSLSocket.recv rejects any flag, so MSG_PEEK is out. Read for
                # real instead -- this is a one-way stream, the browser never
                # sends anything on it, so there is nothing to consume but the
                # close itself. Non-blocking, because a half-arrived TLS record
                # would otherwise stall this thread and hold the camera open.
                try:
                    sock.settimeout(0.0)
                    return sock.recv(4096) == b""
                except (ssl.SSLWantReadError, BlockingIOError):
                    return False          # nothing decodable yet, still there
                finally:
                    sock.settimeout(None)
            return sock.recv(4096, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True


# ---- startup --------------------------------------------------------------


def make_cert():
    """A self-signed certificate for this Pi.

    The microphone is why this exists: browsers only hand it over in a secure
    context, so the page has to be served over HTTPS even on a home network.

    The subjectAltName is the part that matters. Browsers have ignored the
    Common Name for years and will reject a certificate that does not name the
    address you typed, so every way of reaching this Pi goes in the list.
    """
    host = socket.gethostname().split(".")[0]
    names = ["DNS:localhost", "IP:127.0.0.1", f"DNS:{host}", f"DNS:{host}.local"]
    ip = local_ip()
    if ip != "127.0.0.1":
        names.append(f"IP:{ip}")

    argv = ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", CFG.key, "-out", CFG.cert, "-days", "3650",
            "-subj", f"/CN={host}", "-addext", "subjectAltName=" + ",".join(names)]
    print(f"\n  making a certificate for {', '.join(names)}")
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        print("  openssl is not installed:  sudo apt install -y openssl\n")
        return 1
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  could not run openssl: {e}\n")
        return 1
    if r.returncode != 0:
        print(f"  openssl failed:\n{r.stderr.strip()}\n")
        return 1

    try:
        os.chmod(CFG.key, 0o600)           # the key is the whole secret
    except OSError:
        pass
    print(f"    certificate  {CFG.cert}")
    print(f"    private key  {CFG.key}  (never leaves the Pi)")
    print("\n  Start the car and open the https:// address it prints. Your "
          "browser will\n  warn once, because nobody has vouched for this "
          "certificate but itself.\n  To stop the warning -- and on an iPhone, "
          "to let Safari use the microphone --\n  open  /leonida-cert.pem  on "
          "the device and trust it.\n")
    return 0


def tls_context():
    """An SSL context if we have a certificate, else None."""
    if CFG.no_tls:
        return None
    if not (os.path.isfile(CFG.cert) and os.path.isfile(CFG.key)):
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CFG.cert, CFG.key)
    return ctx


class CarServer(ThreadingHTTPServer):
    """Serves HTTPS, and is kind to anyone who arrives over plain HTTP.

    Typing an address without a scheme still gets you http://, and a TLS
    listener answers that with a failed handshake -- which looks exactly like
    the car being down. So peek at the first byte: a TLS handshake starts with
    0x16, and anything else is someone knocking in plaintext, who gets left
    unwrapped so the handler can redirect them.
    """

    tls = None

    def get_request(self):
        sock, addr = super().get_request()
        if self.tls is None:
            return sock, addr
        try:
            # Bounded, because this runs on the accept loop: a client that
            # connects and then says nothing must not hold up everyone else.
            sock.settimeout(5)
            first = sock.recv(1, socket.MSG_PEEK)
            sock.settimeout(None)
            if first == b"\x16":
                sock = self.tls.wrap_socket(sock, server_side=True)
        except (OSError, ValueError, ssl.SSLError) as e:
            try:
                sock.close()
            finally:
                raise OSError(f"handshake from {addr[0]} failed: {e}")
        return sock, addr


def _terminate(signum, frame):
    raise KeyboardInterrupt


def local_ip():
    """Best guess at this machine's LAN address (no packets are actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    global CFG

    # Under systemd stdout is a pipe, and a pipe is block-buffered: without
    # this, motor lines sit in a buffer for minutes and `journalctl -f` shows
    # nothing while the car is plainly moving.
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(
        description="RC car: joystick page, motors, camera and mic, all on the Pi")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--video-device", default="/dev/video0")
    ap.add_argument("--audio-device", default="auto",
                    help='ALSA device, or "auto" to take the first capture card')
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--video-format", default="v4l2",
                    help="ffmpeg input format (avfoundation on a Mac)")
    ap.add_argument("--audio-format", default="alsa",
                    help="ffmpeg input format (avfoundation on a Mac)")
    ap.add_argument("--list-devices", action="store_true",
                    help="show the cameras and mics this machine can see, then exit")
    ap.add_argument("--check-camera", action="store_true",
                    help="grab one frame, report what is in it, then exit")
    ap.add_argument("--fetch-model", action="store_true",
                    help="download the depth-overlay runtime and model, then exit")

    ap.add_argument("--speaker-device", default="auto",
                    help='ALSA playback device, or "auto" for the USB audio dongle')
    ap.add_argument("--speaker-format", default="alsa",
                    help="ffmpeg output format (audiotoolbox on a Mac, "
                         "wav to write a file)")
    ap.add_argument("--no-speaker", action="store_true",
                    help="no horn and no speech; the car stays silent")
    ap.add_argument("--horn-tones", default="440,550",
                    help="horn frequencies in Hz, whole numbers, comma separated")

    ap.add_argument("--cert", default=os.path.join(HERE, "cert.pem"))
    ap.add_argument("--key", default=os.path.join(HERE, "key.pem"))
    ap.add_argument("--make-cert", action="store_true",
                    help="generate a self-signed certificate, then exit")
    ap.add_argument("--no-tls", action="store_true",
                    help="serve plain HTTP even if a certificate exists")
    CFG = ap.parse_args()

    if CFG.list_devices:
        list_devices()
        return

    if CFG.fetch_model:
        sys.exit(fetch_assets())

    if CFG.check_camera:
        sys.exit(check_camera())

    if CFG.make_cert:
        sys.exit(make_cert())

    if CFG.audio_device == "auto":
        CFG.audio_device = detect_audio_device()
    if CFG.speaker_device == "auto":
        CFG.speaker_device = ("none" if CFG.no_speaker
                              else detect_speaker_device())

    global HORN_WAVE
    try:
        tones = [round(float(t)) for t in CFG.horn_tones.split(",") if t.strip()]
    except ValueError:
        tones = []
    if not tones:
        print(f"  bad --horn-tones {CFG.horn_tones!r}; using 440,550")
        tones = [440, 550]
    # Rounded to whole hertz on purpose: a whole second then holds a whole
    # number of cycles, so the loop is seamless and the horn never clicks.
    HORN_WAVE = horn_wave(tones)

    if not CFG.no_camera:
        STREAMS["/stream/video"] = Stream(
            "video", video_argv, jpeg_frames, mjpeg_wrapper, VIDEO_QUEUE,
            f"multipart/x-mixed-replace; boundary={BOUNDARY}")
    if not CFG.no_audio:
        STREAMS["/stream/audio"] = Stream(
            "audio", audio_argv, audio_chunks, mp3_wrapper, AUDIO_QUEUE,
            "audio/mpeg")

    stop_all()
    threading.Thread(target=watchdog, daemon=True).start()

    # systemd stops a service with SIGTERM, and the default handling for that
    # is to die on the spot -- with the motors still energised. Route it into
    # the same shutdown path as Ctrl-C so `systemctl stop` parks the car.
    signal.signal(signal.SIGTERM, _terminate)

    server = CarServer(("0.0.0.0", CFG.port), Handler)
    server.tls = tls_context()
    scheme = "https" if server.tls else "http"

    ip = local_ip()
    print()
    print(f"  Leonida ready{'  [MOCK MODE - no motors]' if MOCK else ''}")
    print(f"    drive it   {scheme}://{ip}:{CFG.port}/")
    print(f"    camera     {CFG.video_device if not CFG.no_camera else 'off'}")
    print(f"    mic        {CFG.audio_device if not CFG.no_audio else 'off'}")
    print(f"    speaker    {CFG.speaker_device if not CFG.no_speaker else 'off'}")
    print(f"    failsafe   motors stop after {DEADMAN:.1f}s of silence")
    print(f"    depth      {'available (browser-side)' if depth_ready() else 'off -- run --fetch-model to enable'}")
    if not server.tls:
        # Worth saying plainly: without this the Speak button cannot work at
        # all, and the reason is not obvious from the browser's error.
        print("    speak      unavailable over http -- run --make-cert")
    print("    Ctrl-C to quit")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        stop_all()
        SPEAKER.hush()
        print("\nMotors stopped.")


if __name__ == "__main__":
    main()

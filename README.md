# Leonida

The whole RC car in one program, running on the Raspberry Pi and nothing else.

Open `http://<pi-ip>:8000` on a phone or laptop on the same network and drive.
The page, the motors, the camera and the mic are all served by a single process,
so a stick movement reaches a motor in about a millisecond.

This replaces the old two-machine setup, where a laptop served the joystick page
and forwarded every command across the network to a motor API on the Pi. There is
no forwarding now, and nothing to keep running on a laptop.

---

## Hardware

Three motors on an H-bridge, wired to these BCM pins:

| Motor | Role | Forward | Backward |
|-------|------|---------|----------|
| 1 | front drive | GPIO 12 | GPIO 13 |
| 2 | rear drive | GPIO 20 | GPIO 21 |
| 3 | steering | GPIO 18 | GPIO 19 |

Plus one USB webcam, which supplies both the video and the microphone.

If a drive motor is wired backwards, tick its invert box in the page's settings
rather than rewiring. If steering goes the wrong way, invert steering there too —
or set `FLIP_MOTOR3 = True` near the top of `leonida.py` to make it permanent.

## Install on the Pi

```bash
sudo apt update && sudo apt install -y ffmpeg python3-gpiozero v4l-utils
```

That is the lot. Everything else is the Python standard library — no virtualenv,
no `pip install`, no `requirements.txt`. Python 3.9 or newer, which covers Pi OS
Bullseye (3.9) and Bookworm (3.11); both are tested.

## Run

```bash
python3 leonida.py
```

It prints the URL to open. Useful flags:

```bash
python3 leonida.py --list-devices          # what cameras and mics can it see?
python3 leonida.py --check-camera          # grab one frame and inspect it
python3 leonida.py --width 1280 --height 720 --fps 15
python3 leonida.py --video-device /dev/video2
python3 leonida.py --audio-device plughw:2,0
python3 leonida.py --no-audio              # camera only
python3 leonida.py --port 8080
```

The mic is found automatically (first ALSA capture card). Pass `--audio-device`
if it guesses wrong; `--list-devices` shows the options.

## Start it on boot

Edit `leonida.service` so `User=` and the paths match your account, then:

```bash
sudo cp leonida.service /etc/systemd/system/ && sudo systemctl enable --now leonida
```

Watch it with `journalctl -u leonida -f`. `systemctl stop leonida` stops the
motors before exiting.

---

## Driving

Drag the joystick, or use **W A S D** / arrow keys. **Space** stops, **M** toggles
audio, the gear icon opens settings.

Motors 1 and 2 always get the same speed and direction, taken from how far the
stick is from the centre in *any* direction — pushed fully left is still full
power. Forward or reverse comes from which half of the pad you are in. Motor 3
steers, scaled by how far the stick is from the vertical centre line.

**Min speed** is the lowest power that actually makes a motor turn. Anything past
the deadzone starts at min and scales to max, so there is no band where the motor
only buzzes. Raise it until the smallest nudge moves the car.

Settings are saved in the browser, per device.

## The failsafe

The car stops itself if it stops hearing from the page for **0.6 seconds** —
tab closed, phone locked, out of Wi-Fi range, browser crashed. The old motor API
held its last value indefinitely, which meant closing the tab mid-throttle left
the car driving.

Two independent things keep it alive, and they are deliberately not the same
mechanism:

- The **page** sends a keepalive every 150 ms. This can only happen while its
  script is really running, so a phone that goes into a pocket stops the car even
  if nothing else notices. Three ticks have to go missing before the motors cut,
  so ordinary timer jitter cannot make the car stutter.
- The **server** sends a WebSocket ping every 5 s. A browser answers that in its
  network stack without running any page script, so a backgrounded tab keeps its
  connection instead of reconnecting every few seconds. This keeps the *socket*
  alive; it deliberately does **not** keep the *motors* alive.

Worth knowing when testing by hand: `curl "http://pi:8000/motor1?power=75"` spins
the motor for 0.6 s and then the failsafe stops it. That is the failsafe working,
not a bug.

## Streams

The camera and mic are started only while someone is watching and shut down a
few seconds after the last viewer leaves, so the webcam light is a reliable
indicator. One capture process feeds everyone, so any number of people can watch
at once. (The few seconds of grace stop a reconnecting browser from restarting
ffmpeg over and over — a webcam takes far longer to open than to serve.)

Video is MJPEG copied straight from the webcam with no transcoding, so the Pi
does almost no work. Audio is MP3, which is the one format that plays as a live
stream in every browser, iPhones included.

### How the page shows the camera

Frames go to the browser over a **WebSocket**, one whole JPEG per binary
message, and into an `<img>` as blob URLs.

This is not the obvious choice, and the obvious choices are why an earlier
version showed nothing at all on iPhones:

- Pointing `<img src>` at a `multipart/x-mixed-replace` stream is the classic
  MJPEG trick. Safari does not render it.
- Reading that stream with `fetch()` and parsing the multipart body yourself
  needs streaming response bodies. Safari only grew those in 14.1, and WebKit
  still treats `multipart/x-mixed-replace` specially at the loader level.

A binary WebSocket has worked since Safari 6 and iOS 6, needs no parsing in the
page, and pushes frames rather than waiting to be asked for them.

If a WebSocket cannot be opened at all, the page falls back to reloading
`/snapshot.jpg` — nothing but an `<img>` loading an ordinary JPEG, which works
in anything that can display a picture. It is a real feed, not a token gesture:
roughly 15 frames a second on a LAN, off the same single capture.

`/stream/video` is still served as ordinary MJPEG, so `vlc http://pi:8000/stream/video`
and similar tools keep working. The page just no longer depends on it.

## Endpoints

| Route | Purpose |
|-------|---------|
| `GET /` | the joystick page |
| `GET /ws` | WebSocket — the normal control path |
| `GET /ws/video` | WebSocket — one whole JPEG per binary message; how the page watches |
| `GET /snapshot.jpg` | a single JPEG; the page's fallback, and handy on its own |
| `GET /drive?m1=&m2=&m3=` | all three motors in one request (HTTP fallback) |
| `GET /motor1?power=-100..100` | one motor; also POST with `{"power": -40}` |
| `GET /ping` | refreshes the failsafe on the fallback path |
| `GET /stream/video` | MJPEG, for VLC and other tools |
| `GET /stream/audio` | MP3 |
| `GET /health` | JSON status: motor state, viewers, time since last command |

Control normally goes over the WebSocket: one open connection, one small message
per stick update, all three motors at once. If the socket will not open or drops
mid-drive, the page falls back to `/drive` automatically and keeps retrying the
socket, so a flaky network degrades instead of failing.

---

## Troubleshooting

**"no camera — …"** on the page. The message is ffmpeg's own. Check
`python3 leonida.py --list-devices`, and that nothing else holds the camera
(`sudo fuser -v /dev/video0`). A webcam that does not do MJPEG needs
`--video-format v4l2` with a lower resolution, or it will be rejected outright.

**No audio.** Confirm the device with `arecord -l` and pass it explicitly.
Some webcams enumerate their mic on a different card from the video.

**Status dot stays red.** The page is not getting acks. Check the Pi is
reachable and look at `journalctl -u leonida -f` for control socket lines.

**The feed says "snapshot mode".** The page could not open a video WebSocket and
has fallen back to polling JPEGs. It works, but something is blocking WebSockets
— usually a proxy between the phone and the Pi. Connect to the Pi directly.

**A broken-image icon where the picture should be** (a blue question mark in
Safari), or the page saying frames arrive but cannot be decoded. The transport
is fine and the frames themselves are the problem. Run:

```bash
python3 leonida.py --check-camera
```

It grabs one frame, lists what is inside it, and saves it to
`/tmp/leonida-frame.jpg`. Open that file: if it looks right, the capture is fine
and the trouble is in the browser; if it looks wrong, it is the camera.

Two things it checks for specifically:

- **No Huffman tables.** Most USB webcams leave them out of every frame — the
  AVI1 convention, where the standard tables are assumed. ffmpeg reads such
  frames happily, which is why everything looks healthy on the Pi, but a strict
  decoder is entitled to refuse them. The standard tables are put back
  automatically, and `--check-camera` says so when it happens.
- **Progressive JPEG.** Rare from a webcam, and not reliably displayable as a
  live feed. Try a different `--width`/`--height`.

**Motors twitch but do not turn.** Raise **min speed** in settings.

**It runs but no motors move, and the log says mock mode.** `gpiozero` did not
import — install `python3-gpiozero`. Mock mode is deliberate so the page and
streams can be developed on a laptop.

## Developing on a laptop

Without GPIO it runs in mock mode and prints what it would have driven. ffmpeg's
input format is a flag, so the camera path can be exercised with a Mac webcam or
a synthetic source:

```bash
python3 leonida.py --video-format avfoundation --video-device 0 --no-audio
python3 leonida.py --video-format lavfi --video-device "testsrc=size=640x480:rate=30" --no-audio
```

## Getting it onto the Pi

The repository is private, so the Pi needs credentials. Either authenticate once:

```bash
sudo apt install -y gh && gh auth login          # then: git clone <repo-url>
```

or add a read-only deploy key, which is better for a machine that lives in a car:

```bash
ssh-keygen -t ed25519 -C "leonida-pi" -f ~/.ssh/id_leonida -N ""
cat ~/.ssh/id_leonida.pub    # add under repo Settings > Deploy keys
git clone git@github.com:BoredxInfinity/Leonida.git
```

Updating later is just `git pull` and `sudo systemctl restart leonida`.

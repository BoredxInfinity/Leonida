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

Plus one USB webcam, which supplies both the video and the microphone, and a
speaker on a USB-to-3.5mm audio dongle if you want the car to make noise.

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
python3 leonida.py --fetch-model           # download the depth model, once
python3 leonida.py --make-cert             # enable https, needed for Speak
python3 leonida.py --speaker-device plughw:1,0
python3 leonida.py --no-speaker            # no horn, no speech
python3 leonida.py --width 1280 --height 720 --fps 15
python3 leonida.py --video-device /dev/video2
python3 leonida.py --audio-device plughw:2,0
python3 leonida.py --no-audio              # camera only
python3 leonida.py --port 8080
```

The mic is found automatically: the first ALSA capture card that names itself a
camera, so the speaker dongle's own (unused) mic input does not win. Pass
`--audio-device` if it guesses wrong; `--list-devices` shows the options.

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

## Horn and Speak

Two hold-down buttons on the left of the page, reachable with the thumb that is
not driving. Keyboard: hold **H** for the horn, **V** to speak.

- **Horn** — a two-tone drone, 440 Hz and 550 Hz together, the interval a real
  car horn uses. Exactly one second of it is worked out when the program starts,
  and because both frequencies are whole numbers that second loops seamlessly,
  so sounding the horn costs the Pi no arithmetic at all. It fades out over a
  few milliseconds on release rather than stopping mid-cycle, which would click.
- **Speak** — your phone's microphone, played out of the car. Audio is captured
  on the browser's audio thread, sent as plain 16 kHz PCM over its own
  WebSocket, and played by the same pipeline as the horn. If both are held at
  once they are mixed.

The horn cannot get stuck on. It stops when you release it, when you press STOP
or space, when the last controller disconnects, and — the one that matters —
when the car stops hearing from the page for 0.6 s, the same deadman that stops
the wheels. Drive out of range mid-blast and it goes quiet on its own.

Set the output with `--speaker-device`; by default it finds the USB audio
dongle in `aplay -l`, preferring it over the Pi's own headphone jack and over
HDMI, which is where a Pi with a monitor attached would otherwise send the
horn. `--horn-tones 350,420` changes the note
(whole numbers of hertz, or the loop clicks) and `--no-speaker` keeps the car
silent.

### Speak needs HTTPS

Browsers only hand over a microphone in a secure context. Over plain `http://`
to an IP address the API is not merely refused, `navigator.mediaDevices` does
not exist at all — so the Speak button disables itself and says so. Everything
else works fine over http.

```bash
python3 leonida.py --make-cert
```

That writes `cert.pem` and `key.pem` next to the program, naming this Pi's
address, hostname and `.local` name, and from then on the car serves HTTPS. Both
files are gitignored; the key is never served by any route.

The first visit warns, because nothing has vouched for this certificate but
itself — click through and it works. To stop the warning, and **on an iPhone to
let Safari use the microphone at all**, open `/leonida-cert.pem` on the device
and trust it (on iOS: install the profile, then enable it under Settings →
General → About → Certificate Trust Settings).

If the Pi's address changes, the certificate no longer matches and the warning
comes back — give it a static DHCP lease, or use the `.local` name.

Arriving on `http://` after switching to HTTPS does not hang: the car notices
the connection is not a TLS handshake and redirects you.

## Depth overlay (optional)

A monocular depth model can be laid over the camera feed — near things red, far
things blue. Turn it on under Settings.

**It runs in your browser, not on the car.** That is the whole design. The Pi
holds the GPIO pins and cuts the motors after 0.6 s of silence, so spending its
four cores on a neural network is a good way to make the car stutter to a halt
mid-drive. The frames have already arrived in the browser to be displayed, so
running the model there costs the car nothing at all, works out to one model per
viewer, and can be changed without touching the vehicle.

One-time setup on the Pi (about 70 MB, kept out of git):

```bash
python3 leonida.py --fetch-model
```

That downloads ONNX Runtime Web and Depth Anything V2 Small into `assets/`, and
the Pi serves them itself — so the overlay still works parked somewhere with no
internet, or with the Pi as its own hotspot.

### It needs WebGPU

Measured on a laptop, one 336×252 frame of this model costs:

| | per frame | |
|---|---|---|
| WebGPU, fp16 | **25 ms** | 40 fps |
| WebGPU, int8 | 1578 ms | WebGPU has no int8 kernels for these ops and quietly falls back to the CPU |
| WASM, int8, 1 thread | 1243 ms | |

So this is WebGPU or nothing, and the fp16 model is the one that ships. On a
browser without WebGPU the overlay refuses to start and says so rather than
pretending. End to end — JPEG decode, inference, colouring — expect somewhere
around 15–30 fps on a recent phone or laptop, and nothing at all on an old one.

Two things it will not do. The output is **relative** inverse depth, not metres:
good for "that is nearer than that", useless for "that wall is 2.3 m away".
And the model has no idea what your car is; it is a general-purpose depth
estimator being pointed at a floor.

### It gets out of the way

Inference runs in a Web Worker, so it never blocks the page's main thread —
which matters, because that thread also sends the keepalive that stops the car
from cutting out. Frames are offered to the model only when it is idle; if it
cannot keep up, frames are dropped rather than queued, so the picture stays live
and only the overlay slows down.

As a backstop the page times its own keepalive tick. If that starts arriving
late while the overlay is on, the overlay switches itself off and says so.
Driving comes first.

## Endpoints

| Route | Purpose |
|-------|---------|
| `GET /` | the joystick page |
| `GET /ws` | WebSocket — the normal control path |
| `GET /ws/video` | WebSocket — one whole JPEG per binary message; how the page watches |
| `GET /snapshot.jpg` | a single JPEG; the page's fallback, and handy on its own |
| `GET /assets/<file>` | the depth runtime and model, if fetched (cached hard) |
| `GET /drive?m1=&m2=&m3=` | all three motors in one request (HTTP fallback) |
| `GET /motor1?power=-100..100` | one motor; also POST with `{"power": -40}` |
| `GET /ws/mic` | WebSocket — 16 kHz PCM from a phone, played by the car |
| `GET /horn?on=1` | the horn, for the HTTP fallback path |
| `GET /ping` | refreshes the failsafe on the fallback path |
| `GET /leonida-cert.pem` | this Pi's certificate, for trusting on a phone |
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

**The horn plays somewhere unexpected**, or not at all. `--list-devices` shows
the playback devices; pass the right one with `--speaker-device`. Check the USB
dongle is seated -- unplugged, the guess falls back to the Pi's own jack or, on
a Pi with a monitor, HDMI.

**The Speak button is greyed out.** The page is not on HTTPS. Run
`--make-cert` and use the `https://` address it prints.

**Speak works on Android but not on the iPhone.** iOS is stricter about
self-signed certificates than clicking through the warning covers. Open
`/leonida-cert.pem` on the phone, install the profile, then turn it on under
Settings → General → About → Certificate Trust Settings.

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

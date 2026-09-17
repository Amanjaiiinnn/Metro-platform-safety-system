#!/usr/bin/env python3
"""
mjpeg_relay.py
--------------
A tiny TCP relay, to be run on the HOST MACHINE -- not on the board.

Why this exists
===============
The i.MX93 is cabled to a host over a connection-sharing link and ends up
on 10.42.0.0/24, whose only neighbour is the host at 10.42.0.1. A phone
camera sitting on the office WiFi (e.g. 10.227.155.26) is on a different
network entirely, with no route between the two -- the board gets
"Destination Host Unreachable" no matter how the capture pipeline is
configured. No camera-side setting can fix that; the packets have nowhere
to go.

The host, however, is on BOTH: the shared link to the board, and the
network the phone is on. So it can forward for the board. Run this there
and the board gets a URL it can actually reach:

    host $  python3 mjpeg_relay.py 10.227.155.26:4747
    board$  curl -sI http://10.42.0.1:4747/video        # now works

then point camera 2 at the host instead of the phone:

    VIDEO_PATH_2 = "http://10.42.0.1:4747/video"        # config.py

This is a plain byte-for-byte TCP relay, so it is not specific to
DroidCam or even to MJPEG -- an rtsp:// camera on the far network works
the same way, as does any other IP camera.

Notes
=====
* Run it on the host, and leave it running. It handles one client at a
  time per connection but accepts many; DroidCam itself only serves one
  client at a time, so in practice only one viewer will get frames.
* --listen defaults to 0.0.0.0 so the board can reach it. Restrict it to
  the shared-link address (--listen 10.42.0.1) if the host is also on an
  untrusted network -- this forwards without any authentication, so don't
  expose it more widely than the board.
* Ctrl-C to stop.
"""

import argparse
import socket
import sys
import threading

BUFFER = 65536


def _pump(src: socket.socket, dst: socket.socket):
    """Shovel bytes one way until either side closes.

    No parsing whatsoever: an MJPEG stream is an open-ended HTTP response
    that never completes, so anything that tried to read a whole
    "message" would block forever. Raw byte forwarding is both simpler
    and the only thing that works for a live stream.
    """
    try:
        while True:
            chunk = src.recv(BUFFER)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        # Half-close so the other direction can still drain, rather than
        # yanking the whole socket out from under the opposite pump.
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _handle(client: socket.socket, addr, target_host: str, target_port: int, verbose: bool):
    upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    upstream.settimeout(10.0)
    try:
        upstream.connect((target_host, target_port))
    except OSError as exc:
        print(f"[relay] {addr[0]} -> {target_host}:{target_port} FAILED: {exc}", flush=True)
        client.close()
        upstream.close()
        return

    upstream.settimeout(None)
    if verbose:
        print(f"[relay] {addr[0]} connected -> {target_host}:{target_port}", flush=True)

    threading.Thread(target=_pump, args=(client, upstream), daemon=True).start()
    _pump(upstream, client)

    client.close()
    upstream.close()
    if verbose:
        print(f"[relay] {addr[0]} disconnected", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Forward a TCP port to a camera the board cannot route to."
    )
    parser.add_argument(
        "target",
        help="Camera address as host:port, e.g. 10.227.155.26:4747",
    )
    parser.add_argument("--listen", default="0.0.0.0",
                        help="Address to listen on (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to listen on (default: same as the target port).")
    parser.add_argument("--quiet", action="store_true", help="Only log failures.")
    args = parser.parse_args()

    if ":" not in args.target:
        parser.error("target must be host:port, e.g. 10.227.155.26:4747")
    target_host, target_port_text = args.target.rsplit(":", 1)
    try:
        target_port = int(target_port_text)
    except ValueError:
        parser.error(f"'{target_port_text}' is not a port number")

    listen_port = args.port or target_port

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((args.listen, listen_port))
    except OSError as exc:
        print(f"[relay] cannot bind {args.listen}:{listen_port}: {exc}", file=sys.stderr)
        return 1
    server.listen(8)

    print(f"[relay] listening on {args.listen}:{listen_port} -> {target_host}:{target_port}",
          flush=True)
    print("[relay] point the board at this machine, e.g. "
          f"http://<this-host-ip>:{listen_port}/video", flush=True)

    try:
        while True:
            client, addr = server.accept()
            threading.Thread(
                target=_handle,
                args=(client, addr, target_host, target_port, not args.quiet),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        print("\n[relay] stopped.", flush=True)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

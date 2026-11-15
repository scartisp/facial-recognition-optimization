"""Log power from a FNIRSI FNB58 USB meter on Windows, to CSV.

Wiring: Pi PSU -> FNB58 USB-C in, FNB58 USB-C out -> Pi, FNB58 micro-USB -> this PC.
Only the IN->OUT path is measured; the PC link carries data only.

Connect in this order: (1) PSU -> meter in, (2) meter out -> Pi, (3) micro-USB -> PC
last. Plugging the PSU in while the micro-USB is already connected hung the meter
(Windows: "USB device not recognized" / Device Descriptor Request Failed) until replugged.
The micro-USB socket is loose: keep the cable from pulling on it.

Protocol (USB HID, 64-byte reports, 100 samples/s, 4 samples per packet) is taken
from github.com/baryluk/fnirsi-usb-power-data-logger, reimplemented on pywinusb
(pure Python, Windows HID API) so no unsigned binaries or drivers are needed.

Usage:
    python bench/power_log.py --out results/raw/power_<label>.csv [--duration 600]
Stop with Ctrl+C (or --duration). Timestamps are Unix epoch seconds (PC clock),
so they can be matched to bench phase timestamps from an NTP-synced Pi.
"""
import argparse
import csv
import queue
import sys
import time

import pywinusb.hid as hid

VID, PID = 0x2E3C, 0x5558
SPS = 100
DT = 1.0 / SPS


def crc8(data):
    # width=8 poly=0x39 init=0x42, no reflection, no xorout (from the reference logger)
    c = 0x42
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x39) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def cmd(second_byte, last_byte):
    # report ID 0 + 64 data bytes
    return [0x00, 0xAA, second_byte] + [0x00] * 61 + [last_byte]


def decode(pkt, t_recv):
    """pkt: 64 bytes starting with 0xAA. Returns list of (t, V, A, temp_C) or None."""
    if len(pkt) < 64 or pkt[0] != 0xAA or pkt[1] != 0x04:
        return None
    if crc8(pkt[1:63]) != pkt[63]:
        return "crc"
    out = []
    for i in range(4):
        o = 2 + 15 * i
        v = int.from_bytes(pkt[o:o + 4], "little") / 100000
        a = int.from_bytes(pkt[o + 4:o + 8], "little") / 100000
        temp = int.from_bytes(pkt[o + 13:o + 15], "little") / 10.0
        out.append((t_recv - (3 - i) * DT, v, a, temp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="CSV path")
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until Ctrl+C")
    args = ap.parse_args()

    devs = hid.HidDeviceFilter(vendor_id=VID, product_id=PID).get_devices()
    if not devs:
        sys.exit("FNB58 not found. Check the micro-USB data cable and close the FNIRSI app.")
    dev = devs[0]
    dev.open()
    q = queue.Queue()
    dev.set_raw_data_handler(lambda raw: q.put((time.time(), raw[1:])))  # drop report ID
    report = dev.find_output_reports()[0]

    def send(c):
        report.set_raw_data(c)
        report.send()

    send(cmd(0x81, 0x8E))
    send(cmd(0x82, 0x96))
    send(cmd(0x82, 0x96))

    n = bad = gaps = 0
    lost = False
    start = time.time()
    last_pkt = start
    next_keepalive = start + 1.0
    try:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_unix", "voltage_V", "current_A", "power_W", "meter_temp_C"])
            print(f"Logging to {args.out} (Ctrl+C to stop)", file=sys.stderr)
            while True:
                now = time.time()
                if args.duration and now - start >= args.duration:
                    break
                if now >= next_keepalive:
                    try:
                        send(cmd(0x83, 0x9E))
                    except hid.HIDError as e:
                        print(f"\nMeter connection lost ({e}). Data up to now is saved.", file=sys.stderr)
                        lost = True
                        break
                    next_keepalive = now + 1.0
                try:
                    t_recv, pkt = q.get(timeout=0.5)
                    last_pkt = t_recv
                except queue.Empty:
                    if time.time() - last_pkt > 2.0:
                        gaps += 1
                        print(f"\nWARNING: no data from meter for {time.time() - last_pkt:.1f} s "
                              "(check the micro-USB cable)", file=sys.stderr)
                    continue
                rows = decode(list(pkt), t_recv)
                if rows == "crc":
                    bad += 1
                    continue
                if not rows:
                    continue
                for t, v, a, temp in rows:
                    w.writerow([f"{t:.3f}", f"{v:.5f}", f"{a:.5f}", f"{v * a:.5f}", f"{temp:.1f}"])
                    n += 1
                if n % 500 < 4:
                    f.flush()
                    print(f"\r{n} samples  {v:.3f} V  {a:.4f} A  {v * a:.3f} W   ", end="", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        t_end = time.time()
        # Like the reference logger: stop keepalives, then keep reading until the meter
        # goes quiet, so it is never left streaming into a closed handle.
        if not lost:
            print("\nDraining meter ...", file=sys.stderr)
            quiet_since = time.time()
            drain_end = time.time() + 5.0
            while time.time() < drain_end and time.time() - quiet_since < 1.5:
                try:
                    q.get(timeout=0.2)
                    quiet_since = time.time()
                except queue.Empty:
                    pass
        try:
            dev.close()
        except Exception:
            pass
    el = t_end - start
    print(f"\nDone: {n} samples in {el:.1f} s ({n / max(el, 1e-9):.0f}/s), {bad} CRC-failed packets, "
          f"{gaps} gap warnings", file=sys.stderr)
    if lost or gaps:
        sys.exit("Run had dropouts: treat it as invalid and rerun.")


if __name__ == "__main__":
    main()

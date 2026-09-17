#!/usr/bin/env python3

import subprocess
import re
import sys

# ======= CONFIGURE THESE =======
STATIC_IP = "10.42.0.2"
NETMASK   = "255.255.255.0"
GATEWAY   = "10.42.0.1"
DNS       = "8.8.8.8"
# ===============================


def run(cmd):
    print("Running:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        sys.exit(result.returncode)

    return result.stdout


# Get available services
output = run(["connmanctl", "services"])

# Find first ethernet service
service = None
for line in output.splitlines():
    match = re.search(r"(ethernet\S+)", line)
    if match:
        service = match.group(1)
        break

if service is None:
    print("No Ethernet service found!")
    sys.exit(1)

print("Ethernet service:", service)

# Configure static IP
run([
    "connmanctl",
    "config",
    service,
    "--ipv4",
    "manual",
    STATIC_IP,
    NETMASK,
    GATEWAY
])

# Configure DNS
run([
    "connmanctl",
    "config",
    service,
    "--nameservers",
    DNS
])

print("\nStatic IP configured successfully.")
print(f"IP      : {STATIC_IP}")
print(f"Gateway : {GATEWAY}")
print(f"DNS     : {DNS}")


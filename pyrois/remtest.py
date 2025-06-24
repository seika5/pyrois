import hivemind
import logging

logging.basicConfig(level=logging.DEBUG)

dht = hivemind.DHT(host_maddrs=['/ip4/0.0.0.0/tcp/8888'], start=True)

print("DHT is running. Press Ctrl+C to quit.")
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Exiting...")

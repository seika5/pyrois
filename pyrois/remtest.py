import hivemind
import logging
logging.basicConfig(level=logging.DEBUG)

dht = hivemind.DHT(
    host_maddrs=['/ip4/0.0.0.0/tcp/8888'],
    announce_maddrs=['/ip4/<your-public-ip>/tcp/8888'],
    start=True
)

input("DHT running. Press Enter to exit...\n")

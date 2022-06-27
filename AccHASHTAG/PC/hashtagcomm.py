import paramiko
from scp import SCPClient
import time

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

# server = "pynq.local"
# port = 22
# user = "xilinx"
# password = "xilinx"


def communicate():
    server = "192.168.2.99"
    port = 22
    user = "xilinx"
    password = "xilinx"

    ssh = createSSHClient(server, port, user, password)
    scp = SCPClient(ssh.get_transport())

    # while True:
    scp.put("weights", "/home/xilinx/", recursive=True)
    time.sleep(1)
    scp.get("/home/xilinx/fpgahashes.pkl")
    time.sleep(1)
    
# communicate()
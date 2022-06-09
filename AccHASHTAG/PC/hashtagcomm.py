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
    server = "acesipu.ucsd.edu"
    port = 22
    user = "nojan"
    password = "Zb9$Og@O+3AHsR5++5Rs"

    ssh = createSSHClient(server, port, user, password)
    scp = SCPClient(ssh.get_transport())

    while True:
        scp.put("weights", recursive=True)
        time.sleep(1)
        scp.get("copy.txt")
        time.sleep(1)
    
# communicate()
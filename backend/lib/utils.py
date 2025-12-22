import os
import socket
import subprocess


def get_free_port(start_port=None, end_port=None):
    if start_port is None:
        start_port = int(os.getenv("TENSORBOARD_START_PORT", 6006))
    if end_port is None:
        end_port = int(os.getenv("TENSORBOARD_END_PORT", 6015))
    port = start_port
    while port <= end_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise ConnectionError("no free ports")


def start_tensorboard(logdir: str, port: int):
    subprocess.Popen(
        [
            "tensorboard",
            f"--logdir={os.path.join(logdir)}",
            f"--port={port}",
            "--host=0.0.0.0",
        ]
    )

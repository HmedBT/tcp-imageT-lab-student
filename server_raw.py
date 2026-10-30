import socket
import struct
import time

import cv2
import numpy as np

PORT = 8080
HEADER_SIZE = 8  # 2 bytes height + 2 bytes width + 4 bytes payload length


def recv_exact(sock, n):
    """Read exactly n bytes from a TCP stream.

    TCP is a byte stream: one recv(n) may return fewer than n bytes,
    so we loop until everything has arrived. Returns None if the peer
    closes the connection before n bytes were received.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:  # connection closed early
            return None
        buf.extend(chunk)
    return bytes(buf)


def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)

    print(f"Server listening on port {PORT}...")
    conn, addr = server_socket.accept()
    print(f"Connection established with {addr}")

    try:
        # Wait for the first byte without consuming it, so the timer measures
        # the transfer itself and not the time the client spends preparing the image.
        if not conn.recv(1, socket.MSG_PEEK):
            print("Client closed the connection without sending data.")
            return
        start_time = time.time()

        # TODO 5: Read the exact header bytes
        header = recv_exact(conn, HEADER_SIZE)
        if header is None:
            print("Error: connection closed before the full header arrived.")
            return

        # TODO 6: Unpack the header (same order and format as the client)
        height, width, payload_length = struct.unpack('!HHI', header)

        print(f"Header received: {width}x{height}, payload = {payload_length} bytes")

        pixels = height * width
        if pixels == 0 or payload_length % pixels != 0:
            print("Error: payload length does not match the image dimensions.")
            return

        # TODO 7: Read the exact payload bytes
        payload = recv_exact(conn, payload_length)
        if payload is None:
            print("Error: connection closed before the full image arrived.")
            return

        # TODO 8: Reconstruct the raw numpy array and reshape it to (height, width, channels)
        channels = payload_length // pixels
        img = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, channels))
        cv2.imwrite('received.png', img)
        print(f"Image reconstructed: shape = {img.shape}, saved as 'received.png'")

        # TODO 9: Network performance metrics
        elapsed = time.time() - start_time
        total_bytes = HEADER_SIZE + payload_length
        received_bytes = HEADER_SIZE + len(payload)

        latency_ms = elapsed * 1000
        # TCP retransmits lost segments, so at application level this is normally 0 %.
        packet_loss = (total_bytes - received_bytes) / total_bytes * 100
        throughput_kbps = (total_bytes / 1024) / elapsed if elapsed > 0 else float('inf')

        print(f"Raw Transfer Complete | Size: {HEADER_SIZE + payload_length} bytes")
        print(f"Metrics | Latency: {latency_ms:.3f} ms | Packet loss: {packet_loss:.2f} % | "
              f"Throughput: {throughput_kbps:.2f} KB/s")

    finally:
        conn.close()
        server_socket.close()


if __name__ == '__main__':
    run_server()
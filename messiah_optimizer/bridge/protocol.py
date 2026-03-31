"""
Communication protocol definitions for Optimizer ↔ Messiah Editor.
"""

from enum import IntEnum
import json
import struct


class MessageType(IntEnum):
    """Message types for the bridge protocol."""
    # Scene operations
    EXPORT_SCENE = 1
    CAPTURE_FRAME = 2

    # Asset push
    IMPORT_TEXTURE = 10
    UPDATE_MATERIAL = 11

    # Control
    HOT_RELOAD = 20
    PING = 30
    PONG = 31

    # Camera sync
    CAMERA_UPDATE = 40
    LIGHT_UPDATE = 41


class Protocol:
    """
    Wire protocol: [4 bytes length][JSON payload]

    All messages are JSON-RPC 2.0 format:
    {
        "jsonrpc": "2.0",
        "method": "method_name",
        "params": { ... },
        "id": 1
    }
    """

    @staticmethod
    def encode(method: str, params: dict = None, msg_id: int = 1) -> bytes:
        """Encode a JSON-RPC request to wire format."""
        request = {
            'jsonrpc': '2.0',
            'method': method,
            'params': params or {},
            'id': msg_id,
        }
        payload = json.dumps(request, ensure_ascii=False).encode('utf-8')
        return struct.pack('<I', len(payload)) + payload

    @staticmethod
    def decode(data: bytes) -> dict:
        """Decode a wire-format message to dict."""
        return json.loads(data.decode('utf-8'))

    @staticmethod
    def encode_response(result, msg_id: int = 1) -> bytes:
        """Encode a JSON-RPC response."""
        response = {
            'jsonrpc': '2.0',
            'result': result,
            'id': msg_id,
        }
        payload = json.dumps(response, ensure_ascii=False).encode('utf-8')
        return struct.pack('<I', len(payload)) + payload

    @staticmethod
    def read_message(sock) -> dict:
        """Read a complete message from a socket."""
        size_data = _recv_exact(sock, 4)
        if not size_data:
            return None
        size = struct.unpack('<I', size_data)[0]
        if size > 100 * 1024 * 1024:  # 100MB sanity limit
            raise ValueError(f"Message too large: {size} bytes")
        payload = _recv_exact(sock, size)
        if not payload:
            return None
        return json.loads(payload.decode('utf-8'))

    @staticmethod
    def write_message(sock, data: dict):
        """Write a message to a socket."""
        payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
        sock.sendall(struct.pack('<I', len(payload)) + payload)


def _recv_exact(sock, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

"""Check HTTP request validation without starting a listener."""
from io import BytesIO
import unittest

from serve import Handler


class LocalRequestTest(unittest.TestCase):
    def accepted(self, host_headers):
        handler = Handler.__new__(Handler)
        handler.raw_requestline = b"GET / HTTP/1.1\r\n"
        handler.rfile = BytesIO(host_headers + b"\r\n")
        handler.wfile = BytesIO()
        handler.client_address = ("127.0.0.1", 44321)
        handler.log_message = lambda *args: None
        return handler.parse_request()

    def test_loopback_and_forwarded_ports(self):
        for host in (b"127.0.0.1:8773", b"localhost:9876", b"[::1]"):
            with self.subTest(host=host):
                self.assertTrue(self.accepted(b"Host: " + host + b"\r\n"))

    def test_requires_one_loopback_host(self):
        for headers in (b"", b"Host: example.org\r\n", b"Host: localhost\r\nHost: localhost\r\n"):
            with self.subTest(headers=headers):
                self.assertFalse(self.accepted(headers))

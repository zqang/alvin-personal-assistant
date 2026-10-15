"""Small offline transport tests for the multi-gigabyte benchmark downloader."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

import download_minicpm_35b as subject


class DownloadTests(unittest.TestCase):
    def test_range_resumes_existing_partial_and_verifies_before_install(self):
        payload = b"GGUF" + bytes(range(256)) * 32
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.headers.get("Range"))
                start = int(self.headers["Range"].removeprefix("bytes=").removesuffix("-"))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
                self.send_header("Content-Length", str(len(payload) - start))
                self.end_headers()
                self.wfile.write(payload[start:])

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                directory = Path(root)
                partial = directory / "model.gguf.partial"
                partial.write_bytes(payload[:997])
                artifact = {"path": "model.gguf", "sizeBytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "sourceURL": f"http://127.0.0.1:{server.server_port}/model.gguf"}
                subject.download(directory, artifact)
                self.assertEqual(requests, ["bytes=997-"])
                self.assertEqual((directory / "model.gguf").read_bytes(), payload)
                self.assertFalse(partial.exists())
                subject.download(directory, artifact, verify_only=True)
                self.assertEqual(len(requests), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_validates_hash_and_magic_even_for_partial_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.gguf.partial"
            payload = b"NOT-a-gguf"
            path.write_bytes(payload)
            artifact = {"path": "model.gguf", "sizeBytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            self.assertFalse(subject.valid(path, artifact))

    def test_rejects_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for relative in ["../escape", "/tmp/escape", "experts/../escape", "experts\\escape", "experts//escape"]:
                with self.assertRaises(ValueError):
                    subject.safe_path(directory, relative)
            (directory / "link").symlink_to(directory.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                subject.safe_path(directory, "link/escape")


if __name__ == "__main__":
    unittest.main()

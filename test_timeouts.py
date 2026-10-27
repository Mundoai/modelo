"""Hung generation / stalled download must not hold the only worker.

Loads the real handler.py with runpod/torch/acestep mocked (no GPU), like test_vram_floor.py.
usage: python3 test_timeouts.py
"""
import http.server, os, sys, tempfile, threading, time, types, unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))


def load_handler(env):
    fake_runpod = types.ModuleType("runpod")
    fake_runpod.serverless = mock.MagicMock()
    torch = types.ModuleType("torch")
    torch.cuda = mock.MagicMock()
    gpu = types.ModuleType("acestep.gpu_config")
    gpu.get_effective_free_vram_gb = mock.MagicMock(return_value=40.0)
    mods = {"runpod": fake_runpod, "torch": torch, "acestep": types.ModuleType("acestep"),
            "acestep.gpu_config": gpu}
    with mock.patch.dict(sys.modules, mods), mock.patch.dict(os.environ, env):
        src = open(os.path.join(HERE, "handler.py")).read()
        src = src.replace("initialize_model()\nexcept", "pass\nexcept")
        ns = {"__name__": "handler_under_test", "__file__": os.path.join(HERE, "handler.py")}
        exec(compile(src, "handler.py", "exec"), ns)
    return ns


class Stall(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "100000")
        self.end_headers()
        self.wfile.write(b"x" * 10)
        self.wfile.flush()
        time.sleep(30)  # then stall mid-body

    def log_message(self, *a):
        pass


class Timeouts(unittest.TestCase):
    def test_hung_generation_returns_fast_and_flags_worker(self):
        ns = load_handler({"GENERATION_TIMEOUT": "1"})
        t0 = time.time()
        with self.assertRaises(Exception):
            ns["_run_with_timeout"](lambda: time.sleep(10))
        self.assertLess(time.time() - t0, 3, "timeout waited for the hung thread")
        self.assertTrue(ns["_GPU_HUNG"])

    def test_normal_generation_unaffected(self):  # positive control
        ns = load_handler({"GENERATION_TIMEOUT": "5"})
        self.assertEqual(ns["_run_with_timeout"](lambda: 42), 42)
        self.assertFalse(ns["_GPU_HUNG"])

    def test_stalled_download_times_out(self):
        ns = load_handler({"DOWNLOAD_TIMEOUT": "1"})
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stall)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        t0 = time.time()
        with tempfile.TemporaryDirectory() as d, self.assertRaises(Exception):
            ns["_download"](f"http://127.0.0.1:{srv.server_address[1]}/a.mp3", os.path.join(d, "a"))
        self.assertLess(time.time() - t0, 5)
        srv.shutdown()


if __name__ == "__main__":
    unittest.main()

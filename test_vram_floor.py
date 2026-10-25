"""VRAM floor: healthy workers do no cleanup; a worker left below the floor is refreshed.

Loads the real handler.py with runpod/torch/acestep mocked (no GPU, no model), and runs the
result through the real runpod SDK run_job to prove refresh_worker -> stopPod.
usage: python3 test_vram_floor.py <path-to-unpacked-runpod-sdk>
"""
import asyncio, os, sys, types, unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SDK = sys.argv.pop(1) if len(sys.argv) > 1 else None


def load_handler():
    fake_runpod = types.ModuleType("runpod")
    fake_runpod.serverless = mock.MagicMock()
    torch = types.ModuleType("torch")
    torch.cuda = mock.MagicMock()
    torch.cuda.is_available.return_value = True
    gpu = types.ModuleType("acestep.gpu_config")
    gpu.get_effective_free_vram_gb = mock.MagicMock()
    mods = {"runpod": fake_runpod, "torch": torch, "acestep": types.ModuleType("acestep"),
            "acestep.gpu_config": gpu}
    with mock.patch.dict(sys.modules, mods):
        src = open(os.path.join(HERE, "handler.py")).read()
        src = src.replace("initialize_model()\nexcept", "pass\nexcept")  # skip model pre-warm
        ns = {"__name__": "handler_under_test", "__file__": os.path.join(HERE, "handler.py")}
        exec(compile(src, "handler.py", "exec"), ns)
    return ns, torch, gpu


class VramFloor(unittest.TestCase):
    def setUp(self):
        self.ns, self.torch, self.gpu = load_handler()
        self.patches = mock.patch.dict(sys.modules, {"torch": self.torch,
                                                     "acestep.gpu_config": self.gpu})
        self.patches.start()
        self.gc = mock.patch("gc.collect").start()

    def tearDown(self):
        mock.patch.stopall()

    def run_handler(self, free_seq, result):
        self.gpu.get_effective_free_vram_gb.side_effect = list(free_seq)
        self.ns["_handle_job"] = lambda job: dict(result)
        return self.ns["handler"]({"id": "j1", "input": {}})

    def test_healthy_worker_does_no_cleanup(self):
        out = self.run_handler([33.1], {"audio_url": "u"})
        self.assertNotIn("refresh_worker", out)
        self.gc.assert_not_called()
        self.torch.cuda.empty_cache.assert_not_called()

    def test_below_floor_but_recovers_after_release(self):
        out = self.run_handler([5.7, 33.0], {"audio_url": "u"})
        self.assertNotIn("refresh_worker", out)
        self.gc.assert_called_once()
        self.torch.cuda.empty_cache.assert_called_once()

    def test_stuck_below_floor_refreshes_worker(self):
        out = self.run_handler([1.1, 1.1], {"audio_url": "u"})
        self.assertTrue(out["refresh_worker"])

    def test_failed_job_still_refreshes_and_keeps_error(self):
        out = self.run_handler([1.1, 1.1], {"error": "Insufficient free VRAM"})
        self.assertEqual(out["error"], "Insufficient free VRAM")
        self.assertTrue(out["refresh_worker"])

    def test_exactly_at_floor_is_healthy(self):
        out = self.run_handler([16.0], {"audio_url": "u"})
        self.assertNotIn("refresh_worker", out)

    def test_no_cuda_never_refreshes(self):
        self.torch.cuda.is_available.return_value = False
        out = self.run_handler([], {"audio_url": "u"})
        self.assertNotIn("refresh_worker", out)

    def test_check_error_never_breaks_job(self):
        self.gpu.get_effective_free_vram_gb.side_effect = RuntimeError("nvml")
        self.ns["_handle_job"] = lambda job: {"audio_url": "u"}
        self.assertEqual(self.ns["handler"]({"id": "j1"}), {"audio_url": "u"})

    @unittest.skipUnless(SDK, "needs unpacked runpod SDK path")
    def test_real_sdk_turns_flag_into_stoppod_and_delivers_output(self):
        sys.path.insert(0, SDK)
        from runpod.serverless.modules import rp_job
        self.gpu.get_effective_free_vram_gb.side_effect = [1.1, 1.1]
        self.ns["_handle_job"] = lambda job: {"audio_url": "u", "error": "x"}
        res = asyncio.run(rp_job.run_job(self.ns["handler"], {"id": "j1", "input": {}}))
        self.assertTrue(res.get("stopPod"))
        self.assertEqual(res["output"], {"audio_url": "u"})
        self.assertEqual(res["error"], "x")


if __name__ == "__main__":
    unittest.main(verbosity=2)

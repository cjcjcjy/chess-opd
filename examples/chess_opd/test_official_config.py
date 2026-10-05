"""Compose the exact official launch configuration without importing its GPU stack."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


class OfficialConfigTest(unittest.TestCase):
    def test_official_launcher_and_hydra_overrides(self):
        upstream = Path(os.environ.get("VERL_DIR", ROOT / "vendor/verl")).resolve()
        if not (upstream / "verl/trainer/config").exists():
            self.skipTest("Run scripts/setup_verl.sh --checkout-only")
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            capture = temp / "argv.json"
            python = temp / "python3"
            python.write_text(f"#!{sys.executable}\n" + """import json, os, sys
from pathlib import Path
if sys.argv[1:3] == ['-m', 'verl.trainer.main_ppo']:
    Path(os.environ['CAPTURE_ARGV']).write_text(json.dumps(sys.argv[3:]))
elif sys.argv[1].endswith('preflight_verl.py'):
    pass  # Configuration test only; data preflight is exercised separately.
else:
    raise SystemExit('Unexpected Python entry point: ' + repr(sys.argv))
""")
            python.chmod(0o755)
            env = {**os.environ, "PATH": str(temp) + os.pathsep + os.environ["PATH"],
                   "VERL_USE_UV": "0", "VERL_DIR": str(upstream), "CAPTURE_ARGV": str(capture),
                   "STUDENT_MODEL": str(temp / "student"), "TEACHER_MODEL": str(temp / "teacher"),
                   "TRAIN_DATA": str(temp / "train.parquet"), "VAL_DATA": str(temp / "dev.parquet"),
                   "OUTPUT_DIR": str(temp / "output")}
            subprocess.run(["bash", str(ROOT / "examples/chess_opd/run_train.sh")],
                           env=env, cwd=ROOT, check=True, capture_output=True, text=True)
            overrides = json.loads(capture.read_text())
            with initialize_config_dir(config_dir=str(upstream / "verl/trainer/config"), version_base=None):
                config = compose(config_name="ppo_trainer", overrides=overrides)
                resolved = OmegaConf.to_container(config, resolve=True)
            self.assertEqual(resolved["trainer"]["total_epochs"], 1)
            self.assertTrue(resolved["trainer"]["use_v1"])
            self.assertEqual(resolved["trainer"]["n_gpus_per_node"], 1)
            self.assertEqual(resolved["distillation"]["n_gpus_per_node"], 1)
            loss = resolved["distillation"]["distillation_loss"]
            self.assertEqual(loss["loss_mode"], "k1")
            self.assertTrue(loss["use_policy_gradient"])
            self.assertFalse(loss["use_task_rewards"])
            self.assertEqual(resolved["data"]["train_files"], str(temp / "train.parquet"))
            self.assertFalse(resolved["data"]["apply_chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(resolved["actor_rollout_ref"]["actor"]["loss_agg_mode"], "token-mean")
            (temp / "resolved.json").write_text(json.dumps(resolved, indent=2))


if __name__ == "__main__":
    unittest.main()

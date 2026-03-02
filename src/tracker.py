import json
from pathlib import Path


class Tracker:
    def __init__(self, observe_cfg: dict, run_dir: Path):
        self.observe_cfg  = observe_cfg
        self.run_dir      = Path(run_dir)
        self.tb_dir       = self.run_dir / 'tb'
        self.metrics_path = self.run_dir / 'metrics.jsonl'
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.enable_jsonl = not observe_cfg.get('disable_jsonl', False)
        self._init_tb()

    def _init_tb(self):
        # Check if TensorBoard is disabled
        if self.observe_cfg.get('disable_tensorboard', False):
            self.writer = None
            return
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        except Exception:
            self.writer = None

    def log_metric(self, name: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)
        
        # Only write to JSONL if not disabled
        if self.enable_jsonl:
            with open(self.metrics_path, 'a') as f:
                f.write(json.dumps({'name': name, 'value': float(value), 'step': step}) + '\n')

    def log_config(self, cfg: dict):
        if self.writer is not None:
            self.writer.add_text('config', json.dumps(cfg, indent=2), 0)

    def finish(self, summary: dict):
        if self.writer is not None:
            self.writer.close()

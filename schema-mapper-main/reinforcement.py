import json
from pathlib import Path

class ReinforcementAgent:
    """
    Very small, persistent reinforcement learner for mapping suggestions.
    Stores counts of accepts/rejects in a JSON file and computes confidence as acceptance rate.
    """
    def __init__(self, stats_path="rl_stats.json"):
        self.path = Path(stats_path)
        if not self.path.exists():
            self._write({})
        self.stats = self._read()

    def _read(self):
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _write(self, obj):
        self.path.write_text(json.dumps(obj, indent=2))

    def update(self, suggestion_key, reward=1):
        s = self._read()
        if suggestion_key not in s:
            s[suggestion_key] = {"accepts":0, "rejects":0}
        if reward >= 1:
            s[suggestion_key]["accepts"] += 1
        else:
            s[suggestion_key]["rejects"] += 1
        self._write(s)
        self.stats = s

    def get_confidence(self, suggestion_key):
        s = self._read()
        if suggestion_key not in s:
            return 0.0
        rec = s[suggestion_key]
        total = rec["accepts"] + rec["rejects"]
        if total == 0:
            return 0.0
        return rec["accepts"] / total

    def get_all_stats(self):
        return self._read()

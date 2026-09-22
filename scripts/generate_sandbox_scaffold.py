#!/usr/bin/env python3
"""Generate the deterministic composition layer for one task sandbox."""
from __future__ import annotations
import argparse, json
from pathlib import Path

APP_SOURCE = '''#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from wsgiref.simple_server import make_server
from sandbox_runtime import ContractRewardAggregator, ContractToolRegistry, ContractUserSimulator, DeclarativeMetricEvaluator, DeclarativeToolCompiler, EpisodeStore, ManifestDataStore, SandboxApplication
from task_impl import TaskHooks
ROOT = Path(__file__).resolve().parent
def _json(path): return json.loads(path.read_text(encoding="utf-8"))
def create_app(*, db_path=None):
    contract = _json(ROOT / "BUILD_CONTRACT.json"); artifacts = contract.get("artifacts", {})
    manifest = artifacts["data_manifest"]; store = EpisodeStore(db_path or ROOT / ".runtime" / "episodes.sqlite3")
    data = ManifestDataStore(manifest, ROOT / manifest["root"], store); hooks = TaskHooks(contract, data, store)
    handlers = {**DeclarativeToolCompiler(data).compile_all(contract.get("tool_implementations", [])), **hooks.business_handlers()}
    registry = ContractToolRegistry(contract.get("tools", []), handlers, noise_tools=contract.get("noise_tools", []), event_recorder=store.event)
    user_manifest = artifacts.get("user_simulation_manifest", {}); user_root = ROOT / user_manifest.get("root", "data/user_simulation")
    sessions = [_json(user_root / item["file"]) for item in user_manifest.get("sessions", [])]
    user = ContractUserSimulator(store, sessions, renderer=hooks.user_renderer())
    metric_evaluator = DeclarativeMetricEvaluator(); aggregator = ContractRewardAggregator(contract.get("metrics", []))
    def reward():
        context = hooks.metric_context(); scores = metric_evaluator.evaluate_all(contract.get("metric_implementations", []), context)
        scores.update(hooks.custom_metric_scores(context, scores)); return aggregator.aggregate(scores)
    def reset(episode): data.reset(episode); user.reset(episode); hooks.reset(episode)
    return SandboxApplication(episode_store=store, tool_registry=registry, observation=hooks.observation, reward=reward, user_turn=user.turn, reset_hook=reset, data_hash=data.data_hash)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--host",default="0.0.0.0"); parser.add_argument("--port",type=int,required=True); args=parser.parse_args()
    with make_server(args.host,args.port,create_app()) as server: server.serve_forever()
if __name__ == "__main__": main()
'''

TASK_IMPL_SOURCE = '''"""Task-specific extension points; platform behavior stays in sandbox_runtime.py."""
from typing import Any, Callable, Mapping
class TaskHooks:
    def __init__(self, contract, data, episode_store): self.contract, self.data, self.episode_store = contract, data, episode_store
    def business_handlers(self) -> dict[str, Callable[[dict[str, Any]], Any]]: return {}
    def observation(self) -> dict[str, Any]: return {"episode_id": self.episode_store.current().episode_id}
    def metric_context(self) -> dict[str, Any]:
        return {"business_state": {name: self.data.table(name) for name in self.data.baseline}, "trajectory": self.episode_store.replay(), "final_agent_response": self.episode_store.get_state("final_agent_response", ""), "observation": self.observation()}
    def custom_metric_scores(self, context: Mapping[str, Any], existing: Mapping[str, float]) -> dict[str, float]: return {}
    def user_renderer(self): return None
    def reset(self, episode) -> None: return None
'''

def generate(root: Path) -> None:
    contract = json.loads((root / "BUILD_CONTRACT.json").read_text(encoding="utf-8"))
    if not isinstance(contract, dict): raise ValueError("BUILD_CONTRACT.json must be an object")
    (root / "app.py").write_text(APP_SOURCE, encoding="utf-8"); (root / "task_impl.py").write_text(TASK_IMPL_SOURCE, encoding="utf-8")
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True); args=parser.parse_args(); generate(args.root.resolve()); return 0
if __name__ == "__main__": raise SystemExit(main())

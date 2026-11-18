"""SimAnvil Trace: Canonical trajectory schema, causal replay verification & pluggable adapters."""

from simanvil.trace.adapters import load_episodes, register_episode_adapter
from simanvil.trace.replay_verify import verify_causal_replay
from simanvil.trace.schema import AnvilRecorder, Episode

__all__ = [
    "AnvilRecorder",
    "Episode",
    "load_episodes",
    "register_episode_adapter",
    "verify_causal_replay",
]

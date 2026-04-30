__all__ = ["DungeonEnv", "ZeldaLikeGame"]


def __getattr__(name: str):
    if name == "DungeonEnv":
        from .envs import DungeonEnv

        return DungeonEnv
    if name == "ZeldaLikeGame":
        from .app import ZeldaLikeGame

        return ZeldaLikeGame
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

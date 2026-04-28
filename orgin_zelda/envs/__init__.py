from .base_env import BaseEnv, ZeldaEnv
from .config import ObservationConfig, ZeldaEnvConfig
from .env51_01 import Room51_Task1_Env
from .env51_02 import Room51_Task1_Combat_Env
from .env58 import Room58_Task_Env
from .env58_02 import Room58_Task2_Env

__all__ = [
	"BaseEnv",
	"ZeldaEnv",
	"ZeldaEnvConfig",
	"ObservationConfig",
	"Room51_Task1_Env",
	"Room51_Task1_Combat_Env",
	"Room58_Task_Env",
	"Room58_Task2_Env",
]
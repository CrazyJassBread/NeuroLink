from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from pyboy import PyBoy
import gymnasium as gym
from gymnasium import spaces
from pyboy.utils import WindowEvent
import torch
import torch.nn.functional as F

# train config
TOTAL_STEPS = 3000000
MAX_STEPS = 1000

# rom file address
ADDR_CUR_HEALTH = 0xDB5A
ADDR_MAX_HEALTH = 0xDB5B
ADDR_RUPEE      = 0xDB5E
ADDR_ROOM_ID    = 0xDBAE
ADDR_KEYS       = 0xDBD0

class BaseEnv(gym.Env, ABC):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, game_file: str, save_file: str, goal_room: int | None = None, render_mode: str | None = None, device=None):
        super().__init__()

        # the training state
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.cur_step = 0
        self.episode = 0
        
        # Init the pyboy emulator
        self.game_file = game_file
        self.save_file = save_file
        self.render_mode = render_mode

        window_mode = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(game_file, sound_emulated=False, window=window_mode)
        if window_mode == "SDL2":
            self.pyboy.set_emulation_speed(1) # set the emulation speed to normal for human rendering

        try:
            with open(save_file, "rb") as f:
                self.pyboy.load_state(f)
        except FileNotFoundError:
            print("No existing save file, starting new game")
        
        # action space
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # observation space
        self.init_image_processing(bucket_boundaries=None, bucket_mapping=None)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(8, 10), dtype=np.uint8
        )

        # the game state, include health, rupee, room id...
        self.max_health = self.read_m(ADDR_MAX_HEALTH)
        self.pre_health = self.read_m(ADDR_CUR_HEALTH)
        self.cur_health = self.pre_health

        self.pre_rupee = self.read_m(ADDR_RUPEE)
        self.cur_rupee = self.pre_rupee

        self.cur_room = self.read_m(ADDR_ROOM_ID)
        self.out_side = 0
        
        # self.visited_rooms: set[int] = set()
        # self.visited_tiles: set[tuple[int, int, int]] = set()  # (room_id, tile_x, tile_y)
        
        # the distance reward
        # self.target_pos = None
        # self.pre_distance = None
        # self.cur_distance = None

    # Gymnasium api - reset, step, render, close
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.cur_step = 0
        self.episode += 1

        with open(self.save_file, "rb") as f:
            self.pyboy.load_state(f)
        self.pyboy.tick(1)

        # reset the game state
        self.cur_room = self.read_m(ADDR_ROOM_ID)
        self.pre_health = self.read_m(ADDR_CUR_HEALTH)
        self.cur_health = self.pre_health
        self.pre_rupee = self.read_m(ADDR_RUPEE)
        self.cur_rupee = self.pre_rupee
        self.out_side = 0

        # the reset hook for extra state
        self._reset_extra(options)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    # the reset hook for child envs to reset extrat state
    def _reset_extra(self, options: dict | None):
        pass
    
    def run_action(self, action: int):
        # send input and release after some ticks to simulate a key press
        # TODO : can be improved by checking the game state to decide when to release
        self.pyboy.send_input(self.valid_actions[action])
        self.pyboy.tick(10)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(10)
    
    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action!"
        self.cur_step += 1

        # update the pre-state
        self.pre_health = self.cur_health
        self.pre_rupee = self.cur_rupee

        self.update_before_action()

        self.run_action(action)

        self.update_after_action()
        # update the common game state
        self.cur_health = self.read_m(ADDR_CUR_HEALTH)
        self.cur_room = self.read_m(ADDR_ROOM_ID)
        self.cur_rupee = self.read_m(ADDR_RUPEE)

        # calculate reward and check termination
        reward, terminated = self.calculate_reward()
        truncated = self.cur_step >= MAX_STEPS

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
    
    def update_before_action(self):
        pass

    def update_after_action(self):
        pass

    def render(self):
        if self.render_mode == "rgb_array":
            return self.pyboy.screen.ndarray
        return None

    def close(self):
        self.pyboy.stop()

    def init_image_processing(self, bucket_boundaries: list[float], bucket_mapping: list[int]):
        # define a 16x16 Gaussian kernel for pooling, the size and sigma can be adjusted based on the task complexity, here we just use a simple one for the key picking task ~
        size = 16
        sigma = 4
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        
        kernel = kernel.astype(np.float32)[None, None, :, :]
        self.gaussian_kernel = torch.from_numpy(kernel).to(self.device)
        
        # XXX define bucket boundaries and mapping, which can be adjusted for more complex task scenarios, here we divide the task into four pixel segments
        if bucket_boundaries is None:
            bucket_boundaries = [30, 90, 190]
            bucket_mapping = [0, 1, 2, 3]
        self.bucket_boundaries = torch.tensor(
            bucket_boundaries, dtype=torch.float32, device=self.device
        )
        self.bucket_mapping = torch.tensor(
            bucket_mapping, dtype=torch.uint8, device=self.device
        )

    def _get_obs(self):
        # return the observation, usually the processed screen

        # raw_screen = self.pyboy.screen.ndarray[:128, :160, 0]
        # screen_tensor = torch.tensor(raw_screen, dtype=torch.float32) 
        # reshaped = screen_tensor.unfold(0, 16, 16).unfold(1, 16, 16)
        # pooled = (reshaped * self.gaussian_kernel).sum(dim=(-1, -2)).clamp(0, 255)
        
        # use conv2d for Gaussian pooling
        raw_screen = self.pyboy.screen.ndarray[:128, :160, 0]
        screen_tensor = torch.from_numpy(raw_screen).to(
            self.device, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (1,1,128,160)

        # 16x16 Gaussian pooling with stride 16 to get an (8,10) feature map, which can capture the key information while reducing the observation size
        pooled = F.conv2d(
            screen_tensor,
            self.gaussian_kernel,
            stride=16,
            padding=0,
        ).squeeze(0).squeeze(0)  # (8,10)
        pooled = pooled.clamp_(0, 255)

        indices = torch.bucketize(pooled, self.bucket_boundaries)
        mapped = self.bucket_mapping[indices]
        
        return mapped.cpu().numpy()

    
    def _get_info(self):
        return {
            "goal": bool(self.check_goal()),
            "room": int(self.cur_room),
        }
    
    @abstractmethod
    def check_goal(self) -> bool:
        # check if the goal is achieved, usually based on room id or some state
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self):
        # calculate the reward based on the current state, and return (reward, terminated)
        raise NotImplementedError

    def _get_pos(self) -> Tuple[int, int]:
        # return the current position of Link
        sprite = self.pyboy.get_sprite(2)
        return sprite.x, sprite.y


    def is_dead(self) -> bool:
        # check if the Link is dead
        return self.read_m(ADDR_CUR_HEALTH) == 0

    def is_hurt(self) -> int:
        if not isinstance(self.cur_health, (int, float)) or not isinstance(self.pre_health, (int, float)):
            return 0
        if self.cur_health < self.pre_health:
            return self.cur_health - self.pre_health
        return 0

    def outside_counter_tick(self, max_out: int = 100) -> bool:
        if self.cur_room != self.goal_room:
            self.out_side += 1
        else:
            self.out_side = 0
        if self.out_side >= max_out:
            self.out_side = 0
            return True
        return False

    # def tile_explore_bonus(self) -> bool:
    #     # the tile explore bonus, only give reward when the agent is in the goal room, and explore a new tile that has not been visited before, which can encourage the agent to explore more in the goal room rather than just stay in one place
    #     if self.cur_room == self.goal_room:
    #         tile_x, tile_y = self._get_tile()
    #         key = (int(self.cur_room), tile_x, tile_y)
    #         if key not in self.visited_tiles:
    #             self.visited_tiles.add(key)
    #             return True
    #     return False

    def rupee_gained(self) -> bool:
        # check if rupee gained since last step, and update pre_rupee if gained
        gained = self.cur_rupee > self.pre_rupee
        if gained:
            self.pre_rupee = self.cur_rupee
        return gained

    def read_m(self, addr: int) -> int:
        return self.pyboy.memory[addr]
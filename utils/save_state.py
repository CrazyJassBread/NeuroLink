# ...existing code...
from matplotlib.pylab import rint
from pyboy import PyBoy
from pynput import keyboard
import time

pyboy = PyBoy("game_state/Link's awakening.gb")
load_state = "game_state/Room_58.state"
save_state = "game_state/Room58_task.state"

try:
    with open(load_state, "rb") as f:
        pyboy.load_state(f)
except FileNotFoundError:
    print("No saved state found")

last_save_state = False
running = True

def on_press(key):
    global last_save_state, running
    try:
        c = getattr(key, "char", None)
        if c:
            c = c.lower()
            if c == "x" and not last_save_state:
                with open(save_state, "wb") as f:
                    pyboy.save_state(f)
                print("Game state saved.")
                last_save_state = True
            if c == "q":
                running = False
                return False  # 停止监听器
    except Exception as e:
        print("键盘处理错误:", e)

def on_release(key):
    global last_save_state
    c = getattr(key, "char", None)
    if c and c.lower() == "x":
        last_save_state = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    for i in range(100000):
        if not running:
            break
        pyboy.tick()
        if i % 100 == 0:
            print(f"Tick: {i}")
            print(f"test memory DB14 {pyboy.memory[0xDB14]}")
            print(f"test memory DB13 {pyboy.memory[0xDB13]}")
            print(f"test memory DB15 {pyboy.memory[0xDB15]}")
            # print(f"test memory DB25 {pyboy.memory[0xDB25]}")
            # print(f"test memory DB27 {pyboy.memory[0xDB27]}")
        time.sleep(0.01)
finally:
    listener.stop()
    pyboy.stop()
# ...existing code...
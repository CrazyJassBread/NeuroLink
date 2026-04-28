## hack_memory.py

用来手动破译游戏内存信息，辅助环境开发

使用方法：
- 仅查看帮助 `python hack_memory.py`
- 读取内存 `python hack_memory.py --state state_name`
- 初始化未知地址扫描 `python hack_memory.py --state state_name scan --init`
- 筛选变化地址 `python hack_memory.py --state state_name scan --filter changed --update-snapshot`
- hunt 未知地址 `python hack_memory.py --state state_name hunt --step-baseline` 
    s 保留变化地址、 i 保留变大地址、 d 保留变小地址、 q 退出
#!/bin/bash

# 获取第一个参数
FUNC=$1

# 检查是否输入了参数
if [ -z "$FUNC" ]; then
    echo "error: please provide func parameter."
    echo "example hunt"
    exit 1
fi

# 根据输入的参数执行不同的 Python 文件
case $FUNC in
    "hunt")
        echo "start chasing the memory ..."
        python utils/hack_memory.py --state game_state/Room_51.state hunt --step-baseline
        ;;
    "help")
        echo ""
        python utils/hack_memory.py
        ;;
    *)
        # 如果输入的参数不在上述选项中
        echo "错误: 无效参数 '$FUNC'。"
        echo "可用选项: hunt, help"
        exit 1
        ;;
esac
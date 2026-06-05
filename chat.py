# chat.py
import sys
from think_twice import ThinkTwiceFramework

def main():
    print("正在加载 Think Twice 引擎...")
    try:
        framework = ThinkTwiceFramework()
    except Exception as e:
        print(f"引擎加载失败: {e}")
        sys.exit(1)
    print("引擎加载完成。\n")

    print("Think Twice 交互式对话（流式输出）")
    print("命令：")
    print("  /exit 或 /quit 退出对话")
    print("  /clear 清空对话历史（保留 system 消息）")
    print("  /help  显示帮助\n")

    # 初始化对话历史，您可以自定义 system 消息
    conversation = [
        {"role": "system", "content": "你是Qwen2.5-0.5B-ThinkTwice，相较于基础模型，在诚实性上有显著提升。"}
    ]

    while True:
        try:
            user_input = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出对话。")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/exit", "/quit"):
                print("退出对话。")
                break
            elif cmd == "/clear":
                # 清空历史，只保留 system 消息
                conversation = [msg for msg in conversation if msg.get("role") == "system"]
                print("对话历史已清空。")
                continue
            elif cmd == "/help":
                print("命令：/exit, /quit, /clear, /help")
                continue
            else:
                print(f"未知命令: {user_input}")
                continue

        # 添加用户消息到历史
        conversation.append({"role": "user", "content": user_input})

        print("助手：", end="", flush=True)

        try:
            # 关键：迭代生成器，流式输出每个 token
            for token in framework.generate(conversation):
                print(token, end="", flush=True)
            print()  # 换行
        except Exception as e:
            print(f"\n生成出错：{e}")
            # 出错时移除最后一条用户消息，避免重复
            if conversation and conversation[-1]["role"] == "user":
                conversation.pop()
            continue

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Harness: the loop -- keep feeding real tool results back into the model.
"""
s01_agent_loop.py - The Agent Loop（最小可用的 coding agent）

【这份代码教你什么】
  agent 的本质 = 一个反复"喂食工具结果"的循环。
  整个流程就 4 步：
      用户消息
        -> 模型回复
        -> 如果模型说"我要用工具"，就执行工具
        -> 把工具结果写回 messages
        -> 继续循环（直到模型说"我说完了"）

【为什么写得这么简单】
  agent 不需要复杂的"规划器""反思器""多智能体"。
  这些花哨架构本质都是在这个 loop 上加调料。
  先把这个 loop 吃透，再加东西。
"""
import os
import subprocess
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────
# readline：让命令行输入支持方向键、历史记录、UTF-8。
# 跟 agent 逻辑无关，看不懂可以完全忽略。
# ─────────────────────────────────────────────────────────────
try:
    import readline

    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
    readline.parse_and_bind("set enable-meta-keybindings on")
except ImportError:
    pass

from anthropic import Anthropic  # Anthropic 官方 SDK，用来调 Claude API
from dotenv import load_dotenv  # 从 .env 文件加载环境变量（API key 等）


# ─────────────────────────────────────────────────────────────
# 配置区：读环境变量、初始化 client
# ─────────────────────────────────────────────────────────────
load_dotenv(override=True)

# 如果用了自定义 BASE_URL（比如代理/中转），就清掉 AUTH_TOKEN
# 避免两个认证方式冲突。普通用户不用关心。
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# system prompt：告诉模型"你是谁、该怎么做事"。
# 这是 agent "性格"的定义，写得不同行为就完全不同。
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to inspect and change the workspace. Act first, then report clearly. Use Chinese"
)

# ─────────────────────────────────────────────────────────────
# 工具定义：告诉模型"你能调用哪些工具，每个工具要传什么参数"
# 这里只给了一个工具：bash。
# 思路：与其给 100 个专用工具（read_file / write_file / git_commit ...），
# 不如给一把万能瑞士军刀（bash）。Claude Code 也是这个思路。
# ─────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the current workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},  # 模型要传一个叫 command 的字符串
            },
            "required": ["command"],
        },
    },
]


# ─────────────────────────────────────────────────────────────
# LoopState：agent 的"全部状态"
# 重点：agent 的"记忆"就是这里的 messages 列表，没有别的魔法。
# 每一轮模型回复、每一次工具结果，都会被 append 进去。
# 模型每次"思考"时，看到的就是这整个列表。
# ─────────────────────────────────────────────────────────────
@dataclass
class LoopState:
    messages: list  # 完整对话历史（agent 的记忆）
    turn_count: int = 1  # 跑了多少轮（仅用于调试观察）
    transition_reason: str | None = None  # 为什么继续 / 为什么停（仅用于调试观察）


# ─────────────────────────────────────────────────────────────
# run_bash：执行模型给出的 shell 命令
# 这是真正"动手做事"的地方 —— 模型只会输出文本，
# 真实世界的改变必须由 harness（这个文件）真的去执行。
# ─────────────────────────────────────────────────────────────
def run_bash(command: str) -> str:
    # 极简的"危险命令拦截"。这只是个安全网，不是真正的沙箱！
    # 生产环境要用 Docker / chroot / 限权用户等真正的隔离手段。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(
            command,
            shell=True,  # 让命令通过 shell 解释（支持管道、重定向）
            cwd=os.getcwd(),  # 在当前目录执行
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True,  # 返回字符串而不是 bytes
            timeout=120,  # 防止命令卡死
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

    # 标准输出 + 错误输出合并返回（模型需要看到错误信息才能修正）
    output = (result.stdout + result.stderr).strip()
    # 截断到 5 万字符，防止单次输出把 context 撑爆
    return output[:50000] if output else "(no output)"


# ─────────────────────────────────────────────────────────────
# extract_text：从模型返回的 content 里只把"纯文本"拼出来
# 模型回复是结构化的，可能包含多个 block：
#   [TextBlock("我先看一下"), ToolUseBlock(bash, "ls"), TextBlock("...")]
# 这个函数只挑出 TextBlock 的文字部分给用户看。
# ─────────────────────────────────────────────────────────────
def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""

    texts = []
    for block in content:
        # getattr(对象, "属性名", 默认值)：安全地取属性，没有就返回默认值
        text = getattr(block, "text", None)
        if text:
            texts.append(text)

    return "\n".join(texts).strip()


# ─────────────────────────────────────────────────────────────
# execute_tool_calls：执行模型这一轮想调的所有工具
# 模型一次可以同时要求调多个工具（并行），这里逐个执行。
# ─────────────────────────────────────────────────────────────
def execute_tool_calls(response_content) -> list[dict]:
    results = []

    for block in response_content:
        # 跳过 TextBlock，只处理 ToolUseBlock
        if block.type != "tool_use":
            continue

        command = block.input["command"]
        print(f"\033[33m$ {command}\033[0m")  # 黄色打印命令，让用户看到 agent 在做什么

        output = run_bash(command)
        print(output[:200])  # 只打印前 200 字符，避免刷屏

        # ⚠️ 关键设计：tool_result 必须带 tool_use_id，和模型发出的 tool_use 配对。
        # 配对错了 API 会直接报错。这是 Anthropic API 的硬性要求。
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            }
        )

    return results


# ─────────────────────────────────────────────────────────────
# run_one_turn：跑"一轮"对话 —— 这是整个 agent 最核心的 20 行
# 返回 True 表示还要继续，False 表示这轮对话结束了
# ─────────────────────────────────────────────────────────────
def run_one_turn(state: LoopState) -> bool:
    # ① 把目前所有消息发给模型
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=state.messages,
        tools=TOOLS,
        max_tokens=8000,
    )

    # 🔍 调试：打印模型返回的原始结构（看完成果可以删掉这 3 行）
    print(f"\n\033[35m=== RAW RESPONSE (turn {state.turn_count}) ===\033[0m")
    print(response.model_dump_json(indent=2))
    print(f"\033[35m=== END ===\033[0m\n")

    # ② 把模型的回复追加到 messages（这就是"记忆"形成的瞬间）
    state.messages.append({"role": "assistant", "content": response.content})

    # ③ 看模型的退出原因：
    #    - "tool_use"  = 模型还想调工具，循环要继续
    #    - 其他值（如 "end_turn"）= 模型说完了，循环结束
    # ⚠️ 关键设计：退出条件是模型自己说"我说完了"，不是写死轮数限制。
    if response.stop_reason != "tool_use":
        state.transition_reason = None
        return False

    # ④ 真的去执行工具，拿到真实结果
    results = execute_tool_calls(response.content)
    if not results:
        state.transition_reason = None
        return False

    # ⑤ 把工具结果追加回 messages，让下一轮模型能看到
    # ⚠️ 反直觉点：tool_result 的 role 是 "user"，不是 "tool"。
    # Anthropic API 规定：工具结果以"用户身份"伪装回传给模型。
    state.messages.append({"role": "user", "content": results})

    state.turn_count += 1
    state.transition_reason = "tool_result"
    return True  # 还有工具要跑，下一轮继续


# ─────────────────────────────────────────────────────────────
# agent_loop：反复跑 run_one_turn，直到它返回 False
# 这就是整个文件的"灵魂"，但代码只有两行。
# ─────────────────────────────────────────────────────────────
def agent_loop(state: LoopState) -> None:
    while run_one_turn(state):
        pass


# ─────────────────────────────────────────────────────────────
# 主程序：命令行交互入口
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ⚠️ 关键设计：history 在 while 循环 *外面* 定义。
    # 这样多轮用户输入之间，messages 是连续累积的，agent 才有"跨轮记忆"。
    # 如果 history 写在 while 里面，每次输入都会清空，agent 就"失忆"了。
    history = []

    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")  # 青色提示符
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        # 把用户输入加到 messages，然后跑 agent loop
        history.append({"role": "user", "content": query})
        state = LoopState(messages=history)
        agent_loop(state)

        # 打印 agent 最后一句话给用户看（中间的工具调用过程已经实时打印了）
        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()

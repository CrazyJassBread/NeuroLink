api: sk-McFpv8jzFDbmoeF5CPdqAArGWRaNAvTeeEvqLAqKBOgETN7z

进入项目目录
cd your-project-folder

windows PowerShell 设置环境变量
$env:ANTHROPIC_BASE_URL = "https://yunwu.ai"
$env:ANTHROPIC_AUTH_TOKEN = "sk-McFpv8jzFDbmoeF5CPdqAArGWRaNAvTeeEvqLAqKBOgETN7z"
$env:API_TIMEOUT_MS = "300000"  # 设置为 300 秒超时

windows cmd 设置环境变量
set ANTHROPIC_BASE_URL=https://yunwu.ai
set ANTHROPIC_AUTH_TOKEN=sk-...
set API_TIMEOUT_MS=300000

启动 Claude Code
claude
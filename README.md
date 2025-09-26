# Claude Data Agent

A sophisticated agentic system built with **PydanticAI** that gives Claude the ability to:
- Write and execute Python scripts in isolated workspaces
- Implement ReAct (Reasoning-Acting-Observing) patterns
- Generate detailed JSON logs of each reasoning stage
- Load configuration from YAML files with environment variable support
- Run completely autonomous data analysis tasks

This project demonstrates both basic agent functionality and advanced ReAct loop implementations with full observability through structured JSON logging.

---

## Quickstart

1) **Python 3.11+** recommended.

2) Install dependencies using uv:

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project and dependencies
uv pip install .
```

3) Set up environment variables in `.env` file:

```bash
# Copy and edit the environment file
cp .env.example .env  # Create if doesn't exist

# Add your API keys to .env:
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...  # Optional, for OpenAI models
```

4) Run the agent with a task:

```bash
# Basic usage (uses config/config.yaml for settings)
python run_agent.py --task "Analyze the sample data and report a basic statistic."

# Using a task from file
python run_agent.py --task-file ./my_task.txt

# Override model (default: claude-sonnet-4 from config)
python run_agent.py --model "openai:gpt-4o-mini" --task "Your task here"
```

## Configuration

The agent behavior is controlled by `config/config.yaml`:

```yaml
model:
  provider: "anthropic"
  name: "claude-sonnet-4-20250514"

agent:
  max_tools: 20         # Tool call budget
  timeout_sec: 60       # Timeout per tool call
  allow_network: false  # Network access policy

react_demo:
  target_number: 5      # For ReAct counting demo
  max_cycles: 10        # Maximum demo cycles

workspace:
  base_dir: ".runs"     # Where run artifacts are stored
```

---

## How It Works

Each agent run follows this pattern:

1. **Initialization**: Creates isolated workspace in `.runs/{timestamp}-{id}/workspace/`
2. **Task Processing**: Agent receives task and system instructions
3. **Tool Execution**: Calls tools like `write_file`, `run_python`, `read_file`, `list_files`
4. **ReAct Loop**: For complex tasks, implements Reasoning → Acting → Observing cycles
5. **JSON Logging**: Saves detailed logs of each reasoning stage with inputs/outputs
6. **Final Answer**: Returns structured result with confidence score and artifact paths

**Key Features:**
- ✅ **Isolated Execution**: Each run gets its own sandboxed workspace
- ✅ **JSON Observability**: Complete tracing of reasoning stages
- ✅ **Configuration-Driven**: YAML config with dotenv support
- ✅ **Multiple Model Support**: Anthropic Claude, OpenAI GPT models
- ✅ **Budget Controls**: Configurable tool call limits and timeouts

---

## Hello World Exercises

Two validated exercises demonstrate the system capabilities:

### ✅ Exercise 1: Basic Greeting
```bash
python run_agent.py --task "Write a Python script that prints 'Hello! I am your Claude data agent.' when run, then run it and confirm it works."
```

**Tests**: Basic write_file → run_python → read_file workflow

### ✅ Exercise 2: ReAct Counting Demo
```bash
python run_agent.py --task "Create a comprehensive ReAct counting demonstration with JSON logging of all reasoning stages, simulating alternating agent/user turns until reaching the target number."
```

**Tests**:
- Advanced ReAct (Reasoning-Acting-Observing) loop implementation
- JSON logging with structured input/output data
- Config-driven behavior (target_number, max_cycles)
- Autonomous execution without user input

**Output**: Complete JSON logs showing every reasoning stage with timestamps:
- `cycle_01_reasoning.json` - Agent's reasoning process
- `cycle_01_acting.json` - Actions taken with rationale
- `cycle_01_observing.json` - Observations and state updates

## Notes

- Keep tasks modest: simple data munging, small algorithms, tiny CSVs, etc.
- If you truly need third-party libraries, adapt `runner_utils.py` to permit `uv`/`pip` with a whitelist.
- This starter is intentionally small: no DB, no web server, no fancy orchestration. Perfect for a take‑home demo.


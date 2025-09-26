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
- ✅ **Parallel Processing**: Run multiple queries concurrently (5x speedup)
- ✅ **Configuration-Driven**: YAML config with dotenv support
- ✅ **Multiple Model Support**: Anthropic Claude, OpenAI GPT models
- ✅ **Budget Controls**: Configurable tool call limits and timeouts
- ✅ **Real-time Progress**: Console messages and worker tracking
- ✅ **Automated Quality Assessment**: LLM judging with accuracy scoring

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

---

## 🚀 Batch Processing & Parallel Execution

The system includes a powerful batch processing engine for running multiple queries concurrently:

### Basic Batch Execution

```bash
# Run queries from a JSON file (serial execution)
python run_batch_queries.py queries.json --count 5

# Run specific number of queries
python run_batch_queries.py queries.json --count 3
```

### 🔥 Parallel Execution (5x Performance Boost)

```bash
# Parallel execution with 5 workers (recommended)
python run_batch_queries.py queries.json --count 5 --workers 5

# Scale workers based on your needs
python run_batch_queries.py queries.json --workers 3
```

### Query File Format

Create a `queries.json` file:

```json
{
  "queries": [
    {
      "id": "q001",
      "category": "factual",
      "query": "What was the total scheduled quantity in Texas during 2023?"
    },
    {
      "id": "q002",
      "category": "analysis",
      "query": "How many pipeline companies are in the dataset?"
    }
  ]
}
```

### Advanced Features

**🎯 Real-time Progress Tracking**
- Console messages show live progress: `[progress] Loading dataset...`
- Worker identification in parallel mode: `[Worker-1] Processing query...`
- Performance metrics with speedup calculations

**🔍 Automated Quality Assessment**
- Built-in LLM judging with accuracy scores (1-5 scale)
- Cost tracking and optimization
- Code artifact verification for reproducibility

**📊 Comprehensive Logging**
- Complete execution history in `.runs/` directories
- Metadata collection (git status, environment, performance)
- Structured JSON logs for retrospective analysis

### Batch Command Options

```bash
# Full feature set
python run_batch_queries.py queries.json \
  --count 5 \
  --workers 5 \
  --template "templates/data_analysis_agent_prompt.txt" \
  --model "anthropic:claude-sonnet-4-20250514" \
  --max-tools 15 \
  --timeout 300 \
  --output "my_results.json"

# Disable features for faster execution
python run_batch_queries.py queries.json \
  --workers 3 \
  --no-judge \        # Skip LLM judging
  --no-stream \       # Disable live output
  --no-console-updates  # Disable progress messages
```

### Performance Comparison

| Mode | 5 Queries | Wall Clock Time | Speedup |
|------|-----------|-----------------|---------|
| Serial (`--workers 1`) | ~300s | 300s | 1x |
| Parallel (`--workers 5`) | ~300s | ~60s | **5x** |

Perfect for batch analysis, model evaluation, and production data processing workflows.

---

## Notes

- Keep tasks modest: simple data munging, small algorithms, tiny CSVs, etc.
- If you truly need third-party libraries, adapt `runner_utils.py` to permit `uv`/`pip` with a whitelist.
- This starter is intentionally small: no DB, no web server, no fancy orchestration. Perfect for a take‑home demo.


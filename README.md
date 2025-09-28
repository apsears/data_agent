# Claude Data Agent

> **🚀 September 2025 Update:** This project has been completely migrated from PydanticAI to **native Anthropic tool calling** with massive improvements: 86.5% cost reduction, 5/5 accuracy scores, and simplified architecture. The system now includes automatic file versioning and deterministic execution. See [migration docs](docs/2025_09_27_20_45_pydantic_ai_removal_complete.md) for details.

A sophisticated agentic system built with **native Anthropic tool calling** that gives Claude the ability to:
- Write and execute Python scripts in isolated workspaces with automatic versioning
- Implement ReAct (Reasoning-Acting-Observing) patterns with full transparency
- Generate detailed JSON logs of each reasoning stage and tool execution
- Load configuration from YAML files with environment variable support
- Run completely autonomous data analysis tasks with comprehensive audit trails
- Prevent file overwrites through automatic `_v001`, `_v002` versioning
- Execute deterministic batch processing with retry ledger instrumentation

This project demonstrates both basic agent functionality and advanced ReAct loop implementations with full observability through structured JSON logging and comprehensive error handling.

## Key Features

### 🔄 **Automatic File Versioning**
- Prevents file overwrites with automatic `_v001`, `_v002`, etc. suffixes
- Complete audit trail of all script executions
- Tool communicates actual filenames back to agent for proper artifact tracking

### 🔍 **Native Transparent Agent**
- Direct Anthropic API integration with native tool calling
- Enhanced error messages and validation
- Configurable token limits and timeout handling
- Comprehensive debugging and observability

### 📊 **Retry Ledger System**
- Tracks tool execution attempts and outcomes
- Distinguishes between genuine failures and framework issues
- Comprehensive error categorization and analysis

### ⚙️ **Deterministic Execution**
- Resolved nondeterministic failures through token limit optimization
- Enhanced tool schema validation
- Improved error communication between tools and agent

## Examples

See [EXAMPLES.md](EXAMPLES.md) for real query/answer pairs from the data science agent analyzing US natural gas pipeline transportation data.

## Quick Execution:

Install:
```bash
uv lock && uv pip install .
```

then run, e.g.:
```
source .venv/bin/activate && python run_agent.py --task "What was the total scheduled quantity in Texas during 2023?"
```

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
- ✅ **Optimized Tool Calls**: Fused write_file_and_run_python tool reduces overhead
- ✅ **Structured Response Format**: Mandatory response.json with rubric compliance
- ✅ **No-Fallback Design**: Fail-fast approach for reliable error detection
- ✅ **Production Ready**: Recent fixes ensure reliable script execution and logging
- ✅ **Data Analysis Capabilities**: Successfully analyzes multi-GB datasets with comprehensive results
- ✅ **Token Usage & Cost Tracking**: Real-time tiktoken counting with detailed cost breakdowns

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
- Efficiency scoring based on execution time (<60s=5, 60-120s=4, 120-180s=3, 180-240s=2, 240-300s=1, >300s=0)
- Cost tracking and optimization
- Code artifact verification for reproducibility

**💰 Token Usage & Cost Tracking**
- Real-time tiktoken counting during agent execution
- Accurate cost calculation using `config/anthropic_pricing.tsv` and `config/openai_pricing.tsv`
- Detailed cost breakdowns in metadata.json (input tokens, output tokens, total cost)
- Model-specific pricing with support for cached tokens
- Cost aggregation across batch operations
- Combined cost summaries for agent + judging operations

**📊 Comprehensive Logging**
- Complete execution history in `.runs/` directories
- Batch results automatically saved to `results/` folder
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

## 🔧 Troubleshooting

### Common Issues and Solutions

**🔴 Script Execution Errors**
- **Issue**: `[Errno 2] No such file or directory` when running scripts
- **Solution**: Fixed in v2025.09.27 - ensure you have the latest version with proper directory creation

**🔴 JSON Serialization Errors**
- **Issue**: `Object of type datetime is not JSON serializable`
- **Solution**: Fixed in v2025.09.27 - all JSON operations now use `default=str`

**🔴 Missing response.json Fields**
- **Issue**: Agent response missing required rubric fields
- **Solution**: Check template configuration and ensure all required fields are specified

**🔴 Log File Warnings**
- **Issue**: Warning messages about log file creation
- **Solution**: Non-blocking warnings - agent execution continues normally

### Recent Major Updates (September 2025)

✅ **🚀 MAJOR: Migrated from PydanticAI to Native Anthropic Tool Calling** - Complete architecture overhaul
✅ **✨ 86.5% Cost Reduction** - From $0.1061 to $0.0143 per query with improved accuracy
✅ **🎯 Perfect Accuracy Scores** - Improved from 2/5 to 5/5 accuracy with native tool calling
✅ **🔧 Simplified Codebase** - Removed 200+ lines of complex XML parsing logic
✅ **⚡ Real Tool Execution** - Tools now actually execute instead of hallucinating results
✅ **🏗️ Future-Proof Architecture** - Using official Anthropic API patterns
✅ **Fixed script execution blocking issue** - Resolved react_log.jsonl path creation problems
✅ **Fixed JSON serialization errors** - Added proper datetime handling
✅ **Added tiktoken cost tracking** - Real-time token counting with accurate cost calculation

---

## Notes

- Keep tasks modest: simple data munging, small algorithms, tiny CSVs, etc.
- If you truly need third-party libraries, adapt `runner_utils.py` to permit `uv`/`pip` with a whitelist.
- This starter is intentionally small: no DB, no web server, no fancy orchestration. Perfect for a take‑home demo.


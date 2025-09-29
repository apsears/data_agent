# Data Agent

This data science agent takes task queries from the console and produces code and analysis to generate a final answer. It emphasizes formal causal inference
methodology. 

Practically, it runs too long. An earlier version could produce a single-round factual response about a dataset in ~60s, but this has been lost to bloat. There is also regrettably no chat-style interaction.

Future work: it may not be appropriate to build this sort of iterative analysis in a reasonable amount of time. My next step would be to instead execute many pre-baked recipes in parallel, and filter the results.

## 🚀 Quick Start

### 1. Set up the repository:
```bash
# Install dependencies
uv lock && uv pip install .

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Download the dataset:
```bash
python tools/utils/download_dataset.py
```

### 3. Run a query:
```bash
# Basic query example
python transparent_agent_executor.py --task "Using utilization constraint detection, analyze how Winter Storm Elliott affected Transco operations"
```

### Example Complete Run

Here's what a complete analysis looks like:

```bash
(claude_data_agent) user@Mac claude_data_agent % source .venv/bin/activate && python transparent_agent_executor.py --task "Using utilization constraint detection, analyze how Winter Storm Elliott affected Transco operations" --query-id "49aecfd3" --template templates/causal_analysis_agent_prompt.txt --model "openai:gpt-5-mini" --critic-model "openai:gpt-5-mini" --react-explicit --critic --console-updates
Starting analysis with template: templates/factual_analysis_agent_prompt.txt
Workspace: .runs/20250929-073724-080cc42b
Creating explicit ReAct agent (analyst: openai:gpt-5-mini, critic: openai:gpt-5-mini)
Running native transparent agent analysis...
[07:37:24.703] 🧠 LLM CALL: ReAct Iteration 1 - Analyst Reasoning | Model: openai:gpt-5-mini
📝 Created file: 001_scout_analysis_v001.py (requested: 001_scout_analysis.py)
[07:38:04] Running expert causal inference critic for iteration 0...
[07:38:04.708] 🧠 LLM CALL: ReAct Iteration 0 - Critic Evaluation | Model: openai:gpt-5-mini
[07:39:09.510] 🧠 LLM CALL: ReAct Iteration 2 - Analyst Reasoning | Model: openai:gpt-5-mini
📝 Created file: 002_final_analysis_v001.py (requested: 002_final_analysis.py)
[07:40:39] Running expert causal inference critic for iteration 1...
[07:40:39.041] 🧠 LLM CALL: ReAct Iteration 1 - Critic Evaluation | Model: openai:gpt-5-mini
[07:42:09.273] 🧠 LLM CALL: ReAct Iteration 3 - Analyst Reasoning | Model: openai:gpt-5-mini
📝 Created file: 003_improved_analysis_v001.py (requested: 003_improved_analysis.py)
[07:43:23] Running expert causal inference critic for iteration 2...
[07:43:23.597] 🧠 LLM CALL: ReAct Iteration 2 - Critic Evaluation | Model: openai:gpt-5-mini
[07:44:37.628] 🧠 LLM CALL: ReAct Iteration 4 - Analyst Reasoning | Model: openai:gpt-5-mini
✅ Native transparent agent successfully created comprehensive response.json

============================================================
=== FINAL ANSWER ===
============================================================

Summary answer (short):
- Using utilization-constraint detection on Transcontinental Gas Pipe Line Company, LLC (Transco) around Winter Storm Elliott (event date used: 2024-01-16, pre-window 60 days / post-window 30 days):
  - The automated constraint detector flagged 1 Transco day as a utilization-constraint candidate within the window.
  - The event‑study / DiD-style estimate shows a very small peak estimated drop in scheduled throughput of ~‑0.24% at relative day +25. 95% CI ≈ [‑5.54%, +5.37%].
  - Because the pre-trend test is strongly significant (p ≈ 3.3e‑08), the parallel‑trends assumption fails for the matched control set.

============================================================
=== EXECUTION SUMMARY ===
============================================================
Run directory: .runs/20250929-073724-080cc42b
Total execution time: 484.86s

💰 COST BREAKDOWN:
  Model: openai:gpt-5-mini
  Input tokens: 90,729
  Output tokens: 25,649
  Total cost: $0.0740
```

This example shows:
- **Real-time progress tracking** with timestamps and LLM call logging
- **Automatic file versioning** (001_scout_analysis_v001.py, etc.)
- **Expert critic feedback** providing causal inference guidance
- **Comprehensive analysis** including constraint detection and event-study methodology
- **Cost transparency** with token usage and pricing
- **Complete artifacts** saved to timestamped workspace directory

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
- ✅ **Unified Multi-Provider Support**: Seamless integration of Anthropic Claude and OpenAI GPT models via LiteLLM
- ✅ **Budget Controls**: Configurable tool call limits and timeouts
- ✅ **Real-time Progress**: Console messages and worker tracking
- ✅ **Automated Quality Assessment**: LLM judging with accuracy scoring
- ✅ **Optimized Tool Calls**: Fused write_file_and_run_python tool reduces overhead
- ✅ **Structured Response Format**: Mandatory response.json with rubric compliance
- ✅ **No-Fallback Design**: Fail-fast approach for reliable error detection
- ✅ **Production Ready**: Recent fixes ensure reliable script execution and logging
- ✅ **Data Analysis Capabilities**: Successfully analyzes multi-GB datasets with comprehensive results
- ✅ **Token Usage & Cost Tracking**: Real-time tiktoken counting with detailed cost breakdowns
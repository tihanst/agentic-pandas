# Agentic Pandas

A REPL-style agentic loop using an LLM to write pandas code, executing in a sandboxed Jupyter kernel running inside a Docker container. Describe data transformations in plain language (or a markdown file); the agent generates, runs, and iteratively fixes Python/pandas code until the task succeeds, then saves the output in .csv and .xlsx formats and renders the result as an HTML table in your browser.

---

## How It Works

```
                              AGENTIC LOOP

   User Prompt
       │
       ▼
  ┌─────────┐   kernel state + history   ┌─────────────────────────────────┐
  │  Main   │ ─────────────────────────► │  LLM  (via LiteLLM interface)   │
  │  Loop   │ ◄───────────────────────── │  · External API (OpenAI,        │
  └────┬────┘      python code block     │    Anthropic, Together, etc.)   │
       │                                 │  · Local server (e.g. Ollama)   │
       ▼                                 └─────────────────────────────────┘
  ┌─────────────────┐   error?   ┌──────────────────────────────────┐
  │ Jupyter Kernel  │ ─────────► │  Append traceback to history     │
  │ (Docker container)│          │  → retry LLM call                │
  └────────┬────────┘            └──────────────────────────────────┘
           │ success
           ▼
  ┌────────────────────────┐
  │  CSV / XLSX output(s)  │
  │  → HTML render         │
  │  → Browser open        │
  └────────────────────────┘
           │
           ▼
   Follow-up? ──yes──► loop back to top
           │ no
           ▼
   Save conversation history → exit
```

---

## Architecture

```
agentic-pandas/
├── src/
│   └── agentic_pandas/
│       ├── main.py      # Entry point & agentic REPL loop
│       ├── server.py    # MCP server mode (FastMCP — exposes agent as MCP tools)
│       ├── config.py    # Pydantic-settings config (LLMConfig, FilePathConfig)
│       ├── llm.py       # LiteLLM wrapper (external API or local Ollama)
│       ├── logger.py    # Logging setup (stdout/stderr split, configurable level)
│       ├── message.py   # Message pydantic model (role / content)
│       └── prompts.py   # System prompts + STATE_PROBE + LOAD_STATE snippets
├── docker/
│   └── Dockerfile       # Builds the sandboxed Jupyter kernel image
├── pyproject.toml
└── .env                 # Local config (not committed)
```

| Module | Responsibility |
|---|---|
| `main.py` | REPL loop, Docker kernel lifecycle, error retry, HTML delivery |
| `server.py` | MCP server — exposes the agent as FastMCP tools (`start_session`, `pandas_query`, `end_session`, `diagnose_environment`, `diagnose_kernel`) |
| `config.py` | Loads all settings from `.env` via pydantic-settings; auto-injects `LLM_API_KEY` into the provider-specific env var |
| `llm.py` | `completion_call()` — routes to external API (via LiteLLM) or local Ollama |
| `logger.py` | Configures the `agentic_pandas` logger; INFO/DEBUG → stdout, WARNING+ → stderr |
| `message.py` | `Message(role, content)` — typed conversation turn |
| `prompts.py` | System prompts defining LLM behaviour; `STATE_PROBE` (kernel introspection); `LOAD_STATE` (CSV loader) |
| `docker/Dockerfile` | Defines the sandboxed Python/pandas kernel image (`jupyter-pandas-kernel:latest`) |

### Async design

The entire agentic loop is `async`. The key architectural decision is in `execute_and_capture()`, which offloads the blocking `jupyter_client` kernel call to a thread pool via `asyncio.run_in_executor()`. This means the event loop is never blocked while waiting for the kernel to finish executing code — a kernel execution that takes several seconds does not freeze the process.

In MCP server mode (`server.py`) all tool handlers are `async def`, so FastMCP can interleave requests: a slow `pandas_query` kernel execution does not block `diagnose_environment` or other concurrent tool calls.

---

## Output Modes

The `-s` / `--steps` flag switches between two distinct output behaviours:

```
Default mode (-s NOT set)                    Steps mode (-s set)
─────────────────────────────────            ──────────────────────────────────────
LLM saves one final CSV + XLSX to:           LLM saves each intermediate step to:
  output_files/                                output_files/steps/
    final_result_df_<name>_<ts>.csv              STEP_1_<name>_<ts>.csv
    final_result_df_<name>_<ts>.xlsx             STEP_2_<name>_<ts>.csv
                                             Final step also saved as .xlsx
App reads newest CSV → one HTML tab          App reads all CSVs → one HTML tab per step
```

After a successful run, `file_results()` archives all CSVs and HTML files into a named sub-folder keyed to the result filename, keeping the output directories clean for the next run.

---

## Installation

Requires Python 3.12+, [`uv`](https://github.com/astral-sh/uv), and [Docker](https://docs.docker.com/get-docker/).

```bash
git clone https://github.com/tihanst/agentic-pandas
cd agentic-pandas

uv sync

# One-time: build the sandboxed kernel image (pandas, numpy, matplotlib, openpyxl)
docker build -t jupyter-pandas-kernel:latest docker/
```

Kernel ports 5555–5559 are bound to `127.0.0.1` only. The container is stopped automatically on exit or Ctrl-C.

---

## Configuration

Create a `.env` file in the project directory:

```dotenv
# ── External API provider (e.g. Together AI) ───────────────────────────────
PROVIDER=together
LLM_NAME=gptoss120b

# LiteLLM model string — see https://docs.litellm.ai/docs/providers for format
LLM_ENDPOINT=together_ai/openai/gpt-oss-120b

IS_LOCAL=false

# Optional: if omitted, LiteLLM reads the standard provider env var instead
# (e.g. TOGETHER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY …)
LLM_API_KEY=your-api-key-here

TOP_LEVEL_OUTPUT_PATH=./my_project
```

```dotenv
# ── Local Ollama provider ───────────────────────────────────────────────────
PROVIDER=ollama
LLM_NAME=ollama/gpt-oss-120b
IS_LOCAL=true
LLM_ENDPOINT=http://localhost:11434

TOP_LEVEL_OUTPUT_PATH=./my_project
```

`TOP_LEVEL_OUTPUT_PATH` is the only required path. The following structure is created automatically:

```
my_project/
├── output_files/         # CSVs / XLSX written by the LLM
│   └── steps/            # output for steps mode (-s)
├── html_files/           # Rendered HTML result tables
│   └── steps/            # HTML for steps mode (-s)
├── data_input_files/     # Input CSVs (used with -d)
├── markdown_files/       # Markdown prompt files (used with -i)
└── history/              # Saved conversation histories
```

The top-level directory is mounted into the container at `/sandbox/<top_level_name>/`; all paths in generated code use that mount point. Individual paths can be overridden in `.env` (e.g. `INPUT_PATH=/data/my_csv_files`).

### API Keys

Two options — both result in LiteLLM finding the key as a standard provider env var:

1. **Export before running** — `export OPENAI_API_KEY=...` etc. LiteLLM discovers these automatically.
2. **`LLM_API_KEY` in `.env`** — `config.py` injects it into the correct env var on startup based on `PROVIDER`. Mapping: `together`/`together_ai` → `TOGETHER_API_KEY`, `openai` → `OPENAI_API_KEY`, `anthropic` → `ANTHROPIC_API_KEY`, `groq` → `GROQ_API_KEY`, `mistral` → `MISTRAL_API_KEY`, `cohere` → `COHERE_API_KEY`, `huggingface` → `HUGGINGFACE_API_KEY`, `replicate` → `REPLICATE_API_KEY`, `google_gemini` → `GEMINI_API_KEY`.

### LLM Provider Setup

| Mode | `IS_LOCAL` | How it works |
|---|---|---|
| External API (any LiteLLM-supported provider) | `false` | `LLM_ENDPOINT` is passed directly to `litellm.completion()`. See the [LiteLLM provider docs](https://docs.litellm.ai/docs/providers) for the exact string format per provider. |
| Local server | `true` | Routes to Ollama using `LLM_NAME` and `LLM_ENDPOINT` as `api_base`. Other local servers can be added in `llm.py`. |

---

## Usage

```bash
# Interactive — type a prompt, terminate with _END_, follow up or !exit to quit
uv run agentic-pandas

# Pre-load a CSV as initial_data_frame (bare filename or full path — copied in automatically)
uv run agentic-pandas -d my_data.csv
uv run agentic-pandas -d /path/to/anywhere/my_data.csv

# Non-interactive: run a markdown prompt file (bare filename or full path)
uv run agentic-pandas -i prompt.md

# Save each intermediate step as a separate CSV/HTML
uv run agentic-pandas -s

# Compact conversation history after each iteration to reduce token usage
uv run agentic-pandas -c

# Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL — default: WARNING)
uv run agentic-pandas -l DEBUG

# Combinations
uv run agentic-pandas -d /path/to/my_data.csv -i /path/to/my_analysis.md -s
```

On `!exit`, the full conversation is saved as a timestamped `.txt` file in `history/`.

---

## Error Handling

```
LLM returns code block
        │
        ▼
execute_and_capture() runs code in kernel (Docker container)
        │
   ┌────┴────┐
   │  error? │
   └────┬────┘
   yes  │                              no
        ▼                               ▼
  strip ANSI from traceback      read CSV / XLSX output(s)
  re-probe kernel state          render HTML table(s)
  append error as user msg       open in browser
  move partial output files      re-probe kernel state
  → error_<timestamp>/ subdir    await follow-up input
        │                               │
        ▼                        user types !exit?
  retry LLM call                        │ yes
  (up to 10 code-format retries;        ▼
   unlimited execution retries)  save history → exit
```

On error, any files already written to the output directory are moved into an `error_<YYYY-MM-DD_HH-MM-SS>/` subdirectory for inspection. In steps mode these appear under `output_files/steps/` and `html_files/steps/`.

---

## MCP Server Mode

`server.py` exposes the agent as [Model Context Protocol](https://modelcontextprotocol.io/) tools via [FastMCP](https://github.com/jlowin/fastmcp), driveable by any MCP-compatible client (e.g. Claude Desktop).

```bash
uv run agentic-pandas-server
```

| Tool | Description |
|---|---|
| `start_session` | Starts the Docker kernel. Optionally pre-loads a CSV (`datafile` argument). |
| `pandas_query` | Runs a natural-language data analysis query against the active kernel. |
| `end_session` | Stops the kernel and saves conversation history. |
| `diagnose_environment` | Reports which API keys and config values are visible to the server process. |
| `diagnose_kernel` | Runs the Docker container briefly with captured output to debug startup failures. |

Uses the same `.env` configuration as the CLI. Call `start_session` first, then `pandas_query`, then `end_session`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `jupyter-client` | Manages the IPython kernel connection over TCP |
| `ipykernel` | Python kernel backend (runs inside Docker) |
| `litellm` | Unified LLM interface (supports OpenAI, Anthropic, Together, and many others) |
| `pandas` | DataFrame operations in generated code |
| `matplotlib` | Available to generated code for plotting |
| `pydantic` / `pydantic-settings` | Config models and `.env` loading |
| `mcp` / `fastmcp` | MCP server framework (`server.py` mode) |
| `python-dotenv` | `.env` loading for the MCP server process |

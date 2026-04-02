# Agentic Pandas

A REPL-style agentic loop that lets an LLM write and execute pandas code in a live Jupyter kernel. You describe data transformations in plain language (or a markdown file), and the agent generates, runs, and iteratively fixes Python/pandas code until the task succeeds — then renders the result as an HTML table in your browser.

---

## How It Works

```
                              AGENTIC LOOP

   User Prompt
       │
       ▼
  ┌─────────┐   kernel state + history   ┌─────────────────────────────────┐
  │  Main   │ ─────────────────────────► │  LLM  (via LiteLLM gateway)     │
  │  Loop   │ ◄───────────────────────── │  · External API (OpenAI,        │
  └────┬────┘      python code block     │    Anthropic, Together, etc.)   │
       │                                 │  · Local server (e.g. Ollama)   │
       ▼                                 └─────────────────────────────────┘
  ┌─────────────────┐   error?   ┌──────────────────────────────────┐
  │ Jupyter Kernel  │ ─────────► │  Append traceback to history     │
  │   (in-process)  │            │  → retry LLM call                │
  └────────┬────────┘            └──────────────────────────────────┘
           │ success
           ▼
  ┌─────────────────┐
  │  CSV output(s)  │
  │  → HTML render  │
  │  → Browser open │
  └─────────────────┘
           │
           ▼
   Follow-up? ──yes──► loop back to top
           │ no
           ▼
   Save conversation history → exit
```

### Step-by-Step Walkthrough

1. **Startup** — a `jupyter_client` kernel is launched in-process and all configured directories are created.
2. **Input** — the user types a prompt terminated with `_END_`, or a markdown file is passed via `-i`. Optionally a CSV is pre-loaded into the kernel as `initial_data_frame` via `-d`.
3. **State probe** — `STATE_PROBE` (a snippet of introspection code) is executed in the kernel; its JSON output describes every DataFrame, Series, and scalar currently in scope.
4. **LLM call** — the full conversation history plus the current kernel state are sent to the LLM via [LiteLLM](#litellm-the-llm-gateway), which is instructed to reply with **only** a fenced Python code block.
5. **Code extraction & execution** — the code block is stripped from the response and executed in the kernel via `execute_and_capture()`.
6. **Error handling** — if the kernel returns a traceback, it is cleaned of ANSI codes, appended to the conversation as a user message, and the loop retries automatically from step 4.
7. **Output delivery** — on success the agent reads the CSV(s) the LLM wrote to `OUTPUT_PATH`, renders them as scrollable HTML tables, and opens them in the browser.
8. **Follow-up** — the user can continue asking questions; the kernel state is re-probed before each new LLM call so the model always sees up-to-date variable state.

---

## LiteLLM — The LLM Gateway

**[LiteLLM](https://github.com/BerriAI/litellm)** is the unified LLM API layer used by this project. It provides a single `completion()` interface that works across a wide range of model providers — OpenAI, Anthropic, Together AI, Mistral, Cohere, Groq, and many others — without any provider-specific code. This means you can point the agent at virtually any supported model simply by changing your `.env` configuration.

- For **external API calls** (any cloud-hosted model), LiteLLM handles authentication and routing. Check the [LiteLLM provider docs](https://docs.litellm.ai/docs/providers) to find the correct `model` / endpoint string for your chosen provider and set it as `LLM_ENDPOINT` in `.env`.
- For **local LLM servers**, only [Ollama](https://ollama.com/) is currently supported as the local provider. Support for other local servers (e.g. LM Studio, llama.cpp) can be added with a small change to `llm.py`.

---

## Architecture

```
helper/
├── main.py          # Entry point & agentic REPL loop
├── config.py        # Pydantic-settings config (LLMConfig, FilePathConfig)
├── llm.py           # LiteLLM wrapper (external API or local Ollama)
├── logger.py        # Logging setup (stdout/stderr split, configurable level)
├── message.py       # Message pydantic model (role / content)
├── prompts.py       # System prompts + STATE_PROBE + LOAD_STATE snippets
├── markdown_files/  # Example prompts (used with -i flag)
├── pyproject.toml
└── .env             # Local config (not committed)
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | REPL loop, kernel lifecycle, error retry, HTML delivery |
| `config.py` | Loads all settings from `.env` via pydantic-settings |
| `llm.py` | `completion_call()` — routes to external API (via LiteLLM) or local Ollama |
| `logger.py` | Configures the `agentic_pandas` logger; INFO/DEBUG → stdout, WARNING+ → stderr |
| `message.py` | `Message(role, content)` — typed conversation turn |
| `prompts.py` | System prompts defining LLM behaviour; `STATE_PROBE` (kernel introspection); `LOAD_STATE` (CSV loader) |

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
App reads newest CSV → one HTML tab          App reads all CSVs → one tab per step
```

After a successful run, `file_results()` archives all CSVs and HTML files into a named sub-folder keyed to the result filename, keeping the output directories clean for the next run.

---

## Installation

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone <repo>
cd helper
uv sync
```

---

## Configuration

Create a `.env` file in the `helper/` directory:

```dotenv
# LLM provider identifier (e.g. "together", "openai", "anthropic", "ollama")
PROVIDER=together

# Model name — used for local Ollama calls
LLM_NAME=llama3

# LiteLLM endpoint/model string — used for external API calls
# Check https://docs.litellm.ai/docs/providers for the correct format
# for your chosen provider (e.g. "together_ai/mistralai/Mixtral-8x7B-v0.1",
# "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022")
LLM_ENDPOINT=together_ai/meta-llama/Llama-3-70b-chat-hf

# true = local Ollama, false = external API via LiteLLM
IS_LOCAL=false

# Root directory — all subdirectories are derived from this automatically
TOP_LEVEL_OUTPUT_PATH=./my_project
```

`TOP_LEVEL_OUTPUT_PATH` is the only path variable required. On startup the program creates the following subdirectory structure under it automatically:

```
my_project/
├── output_files/    # CSVs written by the LLM
│   └── steps/       # CSVs for steps mode (-s)
├── html_files/      # Rendered HTML result tables
│   └── steps/       # HTML for steps mode (-s)
├── input_files/     # Place input CSV files here (used with -d)
├── markdown_files/  # Markdown prompt files (used with -i)
└── history/         # Saved conversation histories
```

Any individual path can be overridden in `.env` if needed (e.g. `INPUT_PATH=/data/my_csv_files`). Paths not specified are always derived from `TOP_LEVEL_OUTPUT_PATH`.

### First-Time Use with `-d` (Loading a Data File)

> **Important:** On the very first run, the subdirectory tree does not exist yet, so `input_files/` will not be there before you launch. There are two options:
>
> **Option A — let the program create it for you:**
> Run `python main.py` once without `-d` and immediately type `!exit`. The program will create the full directory tree on startup. Then place your CSV in `<TOP_LEVEL_OUTPUT_PATH>/input_files/` and run again with `-d my_data.csv`.
>
> **Option B — run with `-d` anyway:**
> The directories are created at the very start of execution (before the file is read), so even if the first attempt fails because the file is not yet there, the file system will be fully set up by the time the error occurs. Simply place your CSV in `<TOP_LEVEL_OUTPUT_PATH>/input_files/` and re-run.

### API Keys

LiteLLM automatically discovers API keys from your **environment variables** — you do not need to put them in `.env`. Simply export the standard key variable for your provider before running (e.g. `export OPENAI_API_KEY=...`, `export ANTHROPIC_API_KEY=...`). If you prefer to keep everything in `.env`, you can add the key there and it will be picked up as an environment variable when the settings are loaded.

### LLM Provider Setup

| Mode | `IS_LOCAL` | How it works |
|---|---|---|
| External API (any LiteLLM-supported provider) | `false` | `LLM_ENDPOINT` is passed directly to `litellm.completion()`. See the [LiteLLM provider docs](https://docs.litellm.ai/docs/providers) for the exact string format required per provider. |
| Local server | `true` | Currently routes to Ollama using `LLM_NAME` and `LLM_ENDPOINT` as the `api_base`. Other local servers can be added in `llm.py`. |

> **Note:** When `IS_LOCAL=false`, the value of `LLM_ENDPOINT` must be a valid LiteLLM model identifier for your provider. If it is wrong, LiteLLM will raise an error with a message pointing you to check the endpoint and ensure the correct API key environment variable is set.

---

## Usage

```bash
# Interactive mode — type prompts, end each with _END_
python main.py

# Pre-load a CSV as initial_data_frame in the kernel
python main.py -d my_data.csv

# Non-interactive: run a markdown prompt file
python main.py -i prompt.md

# Save each intermediate step as a separate CSV/HTML
python main.py -s

# Compact conversation history to save tokens
python main.py -c

# Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL — default: WARNING)
python main.py -l DEBUG

# Combinations
python main.py -d my_data.csv -i prompt.md -s
```

### Interactive Session

```
Enter in your prompt ending in _END_, or enter in !exit:

Load the sales data from initial_data_frame, group by region,
calculate total and average revenue, and sort descending._END_

# → LLM generates code → kernel executes → browser opens with result table

Enter in your follow-up question followed by _END_ or to quit enter in !exit:

Now filter to only regions with average revenue above 5000._END_

# → kernel state re-probed → LLM sees current DataFrames → iterates

!exit   # saves conversation history and exits
```

### Markdown Prompt Files (`-i`)

Prompts can be written as markdown and stored in `markdown_files/`. This is useful for repeatable or complex multi-step analyses. See `markdown_files/test1.md` and `markdown_files/macd_test.md` for real examples.

```markdown
## Step 1
Load initial_data_frame. Pivot by date (index) and category (columns),
values are sales. Fill nulls with 0.

## Step 2
Calculate a 7-day rolling mean for each category column.

## Step 3
Export the final smoothed pivot table.
```

Run with:
```bash
python main.py -d sales.csv -i my_analysis.md -s
```

---

## Context Window Management

Each LLM call includes the full conversation history (system prompt + all prior user/assistant turns). As a session grows, this consumes more tokens.

**`-c` / `--compact` flag** — after each successful iteration, `reset_reload_context_compact_history()` replaces the accumulated history with:
- The current kernel state (live DataFrame snapshots)
- A concatenation of all prior user queries (not code or tracebacks)

This gives the LLM enough continuity to understand what has been done, without including every generated code block and error trace.

---

## Error Handling Flow

```
LLM returns code block
        │
        ▼
execute_and_capture() runs code in kernel
        │
   ┌────┴────┐
   │  error? │
   └────┬────┘
   yes  │                              no
        ▼                               ▼
  strip ANSI from traceback      read CSV output(s)
  re-probe kernel state          render HTML table(s)
  append error as user msg       open in browser
  archive partial output files   re-probe kernel state
        │                        await follow-up input
        ▼
  retry LLM call
  (up to 10 code-format retries;
   unlimited execution retries)
```

---

## Kernel State Snapshot

Before every LLM call the `STATE_PROBE` snippet runs inside the kernel and emits a JSON summary of all live variables. The LLM receives this to understand what data it already has access to. Example output:

```json
{
  "initial_data_frame": {
    "type": "DataFrame",
    "shape": [300, 7],
    "columns": ["order_id", "customer_id", "order_date", "region", "product_id", "qty", "unit_price"],
    "dtypes": {"order_id": "int64", "order_date": "datetime64[ns]"},
    "head": [{"order_id": 1, "customer_id": 42}],
    "nulls": {"order_id": 0, "order_date": 0}
  }
}
```

---

## Conversation History

On exit (via `!exit` or EOF), the full conversation is saved to `HISTORY_PATH` as a timestamped `.txt` file containing all message contents. This is useful for auditing exactly what the agent did across a session.

---

## Dependencies

| Package | Purpose |
|---|---|
| `jupyter-client` | Manages in-process IPython kernel |
| `ipykernel` | Python kernel backend |
| `litellm` | Unified LLM gateway (supports OpenAI, Anthropic, Together, and many others) |
| `pandas` | DataFrame operations in generated code |
| `matplotlib` | Available to generated code for plotting |
| `pydantic` / `pydantic-settings` | Config models and `.env` loading |

import warnings
warnings.filterwarnings("ignore")

import atexit
import signal
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[2] / ".env")
import time
import datetime
import asyncio
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP
from jupyter_client.blocking.client import BlockingKernelClient

from .main import (
    execute_and_capture,
    execute_csv_load,
    get_kernel_state,
    start_kernel_container,
    ensure_directories,
    strip_ansi,
    extract_code,
    get_time_sorted_file,
    deliver_to_browser,
    save_history,
    archive_error_files,
    to_container_path,
    CSVLoadError,
    CONTAINER_NAME,
    CONNECTION,
)
from .config import LLMConfig, FilePathConfig
from .llm import LLM
from .message import Message
from .prompts import SYSTEM_PROMPT
from .logger import get_logger

logger = get_logger("agentic_pandas.server")

mcp = FastMCP("pandas-agent")

# Module-level state — persists across tool calls for the lifetime of the MCP server process
_state: Dict[str, Any] = {
    "kc": None,
    "proc": None,
    "path_settings": None,
    "conversation_history": [],
    "accumulated_queries": [],
    "system_prompt": None,
    "kernel_state": {},
    "llm": None,
}


def _cleanup() -> None:
    kc = _state["kc"]
    proc = _state["proc"]
    if kc:
        kc.stop_channels()
    result = subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("docker stop failed (rc=%d): %s", result.returncode, result.stderr.strip())
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    _state["kc"] = None
    _state["proc"] = None
    _state["path_settings"] = None
    _state["llm"] = None
    _state["system_prompt"] = None
    _state["kernel_state"] = {}
    _state["conversation_history"].clear()
    _state["accumulated_queries"].clear()


atexit.register(_cleanup)
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))


@mcp.tool()
async def diagnose_environment() -> str:
    """Check what API keys and config values are visible to the MCP server process."""
    import os
    from pathlib import Path

    env_file = Path(__file__).parents[2] / ".env"
    results = [f".env exists: {env_file.exists()}"]

    # Check for known API key env vars without revealing the values
    for var in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TOGETHER_AI_API_KEY",
                "GROQ_API_KEY", "MISTRAL_API_KEY"]:
        val = os.environ.get(var)
        results.append(f"{var}: {'set (' + str(len(val)) + ' chars)' if val else 'NOT SET'}")

    # Check LLMConfig loaded correctly
    try:
        cfg = LLMConfig()
        results.append(f"LLMConfig.provider: {cfg.provider}")
        results.append(f"LLMConfig.llm_name: {cfg.llm_name}")
        results.append(f"LLMConfig.llm_api_key: {'set' if cfg.llm_api_key else 'NOT SET'}")
    except Exception as e:
        results.append(f"LLMConfig error: {e}")

    return "\n".join(results)


@mcp.tool()
async def diagnose_kernel() -> str:
    """Run the kernel container briefly with captured output to diagnose startup failures."""
    import json
    from pathlib import Path

    path_settings = FilePathConfig()
    ensure_directories(path_settings)

    conn_path = Path("/tmp/kernel_connection.json")
    conn_path.write_text(json.dumps(CONNECTION))

    import os, datetime
    try:
        tz_link = os.readlink('/etc/localtime')
        tz_name = tz_link.split('/zoneinfo/')[-1]
    except (OSError, ValueError):
        tz_name = datetime.datetime.now().astimezone().tzname()

    top_level_name = path_settings.top_level_output_path.name
    cmd = [
        "docker", "run", "--platform", "linux/arm64", "--rm",
        "--name", f"kernel-diag-{CONTAINER_NAME}",
        "-e", f"TZ={tz_name}",
        "-v", f"{conn_path}:/tmp/kernel.json:ro",
        "-v", f"{path_settings.top_level_output_path.resolve()}:/sandbox/{top_level_name}:rw",
        "-p", "127.0.0.1:5555:5555",
        "-p", "127.0.0.1:5556:5556",
        "-p", "127.0.0.1:5557:5557",
        "-p", "127.0.0.1:5558:5558",
        "-p", "127.0.0.1:5559:5559",
        "jupyter-pandas-kernel:latest",
        "/tmp/kernel.json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return f"Exit code: {result.returncode}\n\nStdout:\n{result.stdout}\n\nStderr:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Container ran for 10 seconds without exiting — kernel may be starting correctly."
    except Exception as e:
        return f"Error running diagnostic: {e}"


@mcp.tool()
async def start_session(datafile: str = "") -> str:
    """Start the pandas kernel session. Optionally load a CSV file by providing its filename or path."""

    if _state["kc"] is not None:
        return "Session already running. Call end_session first to restart."

    # Verify docker is reachable and the image exists before proceeding
    docker_check = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if docker_check.returncode != 0:
        return f"Docker not accessible: {docker_check.stderr}"

    image_check = subprocess.run(
        ["docker", "image", "inspect", "jupyter-pandas-kernel:latest"],
        capture_output=True, text=True
    )
    if image_check.returncode != 0:
        return "Docker image 'jupyter-pandas-kernel:latest' not found. Build it first."

    llm_settings = LLMConfig()
    path_settings = FilePathConfig()
    ensure_directories(path_settings)

    _state["path_settings"] = path_settings
    _state["llm"] = LLM(
        llm_settings.provider,
        llm_settings.llm_name,
        llm_settings.llm_endpoint,
        llm_settings.is_local,
    )

    try:
        proc = start_kernel_container(path_settings)
        await asyncio.sleep(1)
        kc = BlockingKernelClient()
        kc.load_connection_info(CONNECTION)
        kc.session.key = CONNECTION["key"].encode()
        kc.start_channels()
        kc.wait_for_ready(timeout=30)
    except Exception as e:
        logs = subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, text=True)
        _cleanup()
        return f"Failed to start kernel: {e}\n\nContainer stdout:\n{logs.stdout}\n\nContainer stderr:\n{logs.stderr}"

    _state["proc"] = proc
    _state["kc"] = kc

    system_prompt = SYSTEM_PROMPT.format(
        path=to_container_path(path_settings.output_path, path_settings)
    )
    _state["system_prompt"] = system_prompt
    _state["conversation_history"].append(Message(role="system", content=system_prompt))

    if datafile:
        try:
            await execute_csv_load(kc, path_settings, datafile)
        except (TimeoutError, CSVLoadError) as e:
            _cleanup()
            return f"Kernel started but failed to load CSV: {e}"

    try:
        _state["kernel_state"] = await get_kernel_state(kc)
    except TimeoutError as e:
        _cleanup()
        return f"Kernel started but timed out getting initial state: {e}"

    loaded = f" Loaded '{datafile}'." if datafile else ""
    return f"Session started.{loaded} Ready for queries."


@mcp.tool()
async def pandas_query(prompt: str) -> str:
    """Run a data analysis query against the active pandas kernel session."""

    kc = _state["kc"]
    if kc is None:
        return "No active session. Call start_session first."

    path_settings = _state["path_settings"]
    conversation_history = _state["conversation_history"]
    llm = _state["llm"]

    _state["accumulated_queries"].append(prompt)
    query = f"The current kernel state is:\n\n{_state['kernel_state']}\n\n{prompt}"
    conversation_history.append(Message(role="user", content=query))

    # Inner loop — retries if LLM produces an execution error
    while True:
        payload = [m.model_dump() for m in conversation_history]

        # Retry until LLM returns a well-formed code block
        code_block = None
        for attempt in range(10):
            result = llm.completion_call(payload)
            clean_result = strip_ansi(result["choices"][0]["message"]["content"])
            try:
                code_block = extract_code(clean_result)
                break
            except ValueError:
                if attempt == 9:
                    return "LLM failed to produce a valid code block after 10 attempts."

        conversation_history.append(Message(role="assistant", content=clean_result))

        try:
            res = await execute_and_capture(kc, code_block)
        except TimeoutError as e:
            return f"Kernel timed out during execution: {e}"

        if res["error"] is not None:
            e = res["error"]
            clean_trace = [strip_ansi(x) for x in e["traceback"]]
            full_error = (
                f"Error name: {e['ename']}\n\n"
                f"Error value: {e['evalue']}\n\n"
                f"Traceback:\n{''.join(clean_trace[1:])}"
            )
            logger.warning(full_error)

            try:
                _state["kernel_state"] = await get_kernel_state(kc)
            except TimeoutError as e:
                return f"Timed out getting kernel state after execution error: {e}"

            archive_error_files(path_settings)
            conversation_history.append(Message(
                role="user",
                content=(
                    f"The current kernel state is {_state['kernel_state']}\n\n"
                    f"The previous code generated the following error, please fix it:\n{full_error}\n\n"
                ),
            ))
            continue

        # Success
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        try:
            df = get_time_sorted_file(path_settings)
        except IndexError:
            return "Code executed but no CSV output was found in the output directory."

        html_path = deliver_to_browser(df, path_settings, timestamp)

        try:
            _state["kernel_state"] = await get_kernel_state(kc)
        except TimeoutError as e:
            return f"Execution succeeded but timed out getting updated kernel state: {e}"

        #return f"Done. Results opened in browser. HTML saved to: {html_path}"
        return f"Done. The head of the results are\n\n: {df.head().to_markdown()}"


@mcp.tool()
async def end_session() -> str:
    """Stop the pandas kernel and save conversation history."""

    if _state["kc"] is None:
        return "No active session."

    path_settings = _state["path_settings"]
    conversation_history = _state["conversation_history"]

    if path_settings and conversation_history:
        try:
            save_history(conversation_history, path_settings)
        except Exception as e:
            logger.warning("Failed to save history: %s", e)

    output_path = path_settings.top_level_output_path.resolve() if path_settings else "output directory"
    _cleanup()
    return f"Session ended. Data saved to {output_path}."


def main():
    mcp.run()

if __name__ == "__main__":
    main()

import warnings
import sys
try:
    import readline  # Unix/macOS only — improves stdin editing; not available on Windows
except ImportError:
    pass
import re
import os
import datetime
import json
import shutil
import webbrowser
import argparse
import queue
from typing import List, Dict, Any, Union, NoReturn
from pathlib import Path
import pprint

import pandas as pd
from jupyter_client.manager import KernelManager
from jupyter_client.blocking.client import BlockingKernelClient

from logger import get_logger, set_logger
from config import LLMConfig, FilePathConfig
from llm import LLM
from message import Message
from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_STEPS, STATE_PROBE, LOAD_STATE


class CSVLoadError(Exception):
    pass

warnings.filterwarnings("ignore")
logger = get_logger("agentic_pandas.main")


def acquire_input(client: BlockingKernelClient, manager: KernelManager, path_settings: FilePathConfig, accumulated_user_queries: List[str], file_name: str | None = None) -> str | None:
    
    lines: List[str] = []
    
    # If file option is passed on CLI read it and execute
    if file_name is not None:
        path = Path(path_settings.markdown_path)
        if not path.is_dir():
            Path.mkdir(path.resolve())
        with open(path.resolve() / file_name, encoding="utf-8") as f:
            query = f.read().rstrip()
            accumulated_user_queries.append(query)
            return query    

    # If file option is not passed on CLI read from stdin
    while True:
        query = sys.stdin.readline()

        if query == "":
            break
        
        stripped = query.rstrip("\n")

        if stripped.strip() == "":
            print("Enter in your prompt ending in _END_, or enter in !exit: \n\n")
            continue

        if stripped.lstrip().startswith("!exit"):
            cleanup_and_exit(client, manager, f"Process complete.\nData found in {path_settings.top_level_output_path.resolve()}")
            break

        if stripped.endswith("_END_"):
            lines.append(stripped[:-5])
            parsed_query = "\n".join(lines).strip()
            print(f"\n\nQuery is : {parsed_query}\n\n")
            accumulated_user_queries.append(parsed_query)
            return parsed_query

        lines.append(stripped)

def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)


def extract_code(text: str) -> str:
    if '```python\n' != text[:10]:
        raise ValueError("Incorrect code block start formatting.")
    if '\n```' != text[-4:]:
        raise ValueError("Incorrect code block end formatting")
    
    return text[10:-4]


def get_kernel_state(client: BlockingKernelClient) -> Dict[str, Any]:
    
    try:
        result = execute_and_capture(client, STATE_PROBE)
    except TimeoutError:
        raise

    for output in result['outputs']:
        if output['type'] == 'stream':
            try:
                dat = json.loads(output['text'])
                return {x:y for x,y in dat.items() if x not in {'In', 'Out', 'original_ps1', 'is_wsl'}}
            except json.JSONDecodeError:
                logger.error("Encountered json decoding error")
    return {}


def execute_and_capture(client: BlockingKernelClient, code: str, timeout: int = 30) -> Dict[str, str]:
    
    msg_id = client.execute(code)
    outputs = []
    error = None
    stream_buffers = {}
    
    while True:
        try:
            msg = client.get_iopub_msg(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"kernel timed-out after {timeout} seconds.")
            
        
        msg_type = msg['msg_type']
        content = msg['content']
        
        if msg_type == 'stream':
            name = content['name']
            stream_buffers[name] = stream_buffers.get(name, '') + content['text']
        elif msg_type == 'execute_result':
            outputs.append({'type': 'result', 'data': content['data']['text/plain']})
        elif msg_type == 'display_data':
            outputs.append({'type': 'display', 'data': content['data'].get('text/plain')})
        elif msg_type == 'error':
            error = {'ename': content['ename'], 'evalue': content['evalue'],
                     'traceback': content['traceback']}
        elif msg_type == 'status' and content['execution_state'] == 'idle':
            break  # kernel finished
    
    for name, text in stream_buffers.items():
        outputs.append({'type':'stream', 'name':name, 'text':text})
    
    return {'outputs': outputs, 'error': error}


def get_time_sorted_file(path_settings: FilePathConfig) -> pd.DataFrame:

    directory = Path(path_settings.output_path).resolve()
    files = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix == '.csv'], key=os.path.getmtime)
    
    # Select last which is newest        
    df = pd.read_csv(files[-1].resolve(), encoding='utf-8', index_col=0)
    return df 


def get_all_time_sorted_files(path_settings: FilePathConfig) -> List[pd.DataFrame]:

    directory = Path(path_settings.output_path).resolve() 
    files = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix == '.csv'], key=os.path.getmtime)

    # Oldest to newest        
    dfs = [pd.read_csv(x.resolve(), encoding='utf-8', index_col=0) for x in files]
    
    return dfs 


def deliver_to_browser(df: pd.DataFrame, path_settings: FilePathConfig, timestamp: str, counter: int | None = None, open_browser: bool = True) -> Path:
    html = f"""
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: monospace; padding: 20px; }}
        .wrapper {{
            overflow: auto;
            max-height: 90vh;
            max-width: 100vw;
            border: 1px solid #ccc;
        }}
        table {{ border-collapse: collapse; white-space: nowrap; }}
        th, td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: left; }}
        th {{ background: #f4f4f4; position: sticky; top: 0; z-index: 1; }}
    </style>
    </head>
    <body>
    <div class="wrapper">
        {df.to_html(index=True)}
    </div>
    </body>
    </html>
    """
     
    path = Path(path_settings.html_path).resolve()

    if not path.is_dir():
        Path.mkdir(path)
    logger.info(f" HTML path is : {path}")

    if counter is None:
        path = path / f"{timestamp}.html"
    else:
        path = path / f"{timestamp}_{counter}.html"

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    if open_browser:
        logger.info(f"Opening HTML file at {path}")
        webbrowser.open(path.as_uri())

    return path


def execute_csv_load(client: BlockingKernelClient, path_settings: FilePathConfig, filename: str):
    path = Path(path_settings.input_path).resolve()
    if not path.is_dir():
        Path.mkdir(path)
    x = execute_and_capture(client, LOAD_STATE.format(filepath=str(path), file=filename))
    if x['error']:
        clean_traceback = [strip_ansi(y) for y in x['error']['traceback']]
        raise CSVLoadError(f"Failed to load CSV. Traceback:\n{''.join(clean_traceback)}")


def archive_error_files(path_settings: FilePathConfig) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for dir_str in (path_settings.output_path, path_settings.html_path):
        directory = Path(dir_str).resolve()
        if not directory.is_dir():
            continue
        files = [f for f in directory.iterdir() if f.is_file()]
        if not files:
            continue
        error_dir = directory / f"error_{timestamp}"
        error_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.move(str(f), error_dir / f.name)
        logger.info(f"Archived {len(files)} Error file(s) to {error_dir}")


def save_history(conversation_history: List[Message], path_settings: FilePathConfig) -> None:
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(path_settings.history_path)
    if not path.is_dir():
        Path.mkdir(path)

    with open(path / f"{timestamp}_history.txt", 'w', encoding='utf-8') as f:
        f.write("\n\n".join([x.content for x in conversation_history[1:]])) # Do not write the SYSTEM message 

def file_results(path_settings: FilePathConfig, timestamp: str = "") -> Path:

    def _helper_file_results(ref_folder: Path) -> Path:
        # Order is oldest to most recent file created
        files: List[Path] = sorted([f for f in  ref_folder.iterdir() if f.is_file()], key=os.path.getmtime)
        if not files:
            return ref_folder
        foldername = ref_folder / files[-1].stem
        foldername.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.move(f, foldername)
        return foldername

    #handle csv
    ref_folder = Path(path_settings.output_path).resolve()
    _ = _helper_file_results(ref_folder)
    #handle html
    ref_folder = Path(path_settings.html_path).resolve()
    html_final_folder = _helper_file_results(ref_folder)
    logger.info(f"\n\nHTML steps folder is: {html_final_folder}\n\n")
    return html_final_folder 

    
def reset_reload_context_compact_history(conversation_history: List[Message], state: Dict[str, Any], path_settings: FilePathConfig, query: str, system_prompt: str, accumulated_user_queries: List[str]) -> None:
    
    #Clear and save conversation state to clean up context window 
    conversation_history.append(Message(role='user', content=f"The ending kernel state is: \n\n{state}\n\n"))
    save_history(conversation_history, path_settings)

    conversation_history.clear()
    conversation_history.append(Message(role='system', content=system_prompt))

    #Test context window compression by adding in SYSTEM plus concatenation of summary of all previous user prompts (not messages as those contain code and error traces)
    
    query = f"The current kernel state is: \n\n{state}\n\n\nThis was achieved through the following user directions:\n{'\n'.join(accumulated_user_queries)}\n\nThe next user instruction is:\n\n" + query
    conversation_history.append(Message(role='user', content=query))
    logger.info(f"Compact conversation history is: {conversation_history[-1].content}")

def ensure_directories(path_settings: FilePathConfig) -> None:
    for path_str in (
        path_settings.output_path,
        str(Path(path_settings.output_path) / 'steps'),
        path_settings.html_path,
        str(Path(path_settings.html_path) / 'steps'),
        path_settings.markdown_path,
        path_settings.history_path,
        path_settings.input_path,
    ):
        Path(path_str).resolve().mkdir(parents=True, exist_ok=True)


def cleanup_and_exit(kc: BlockingKernelClient, km: KernelManager, message: str | None = None,) -> NoReturn:
    if km.is_alive():
        km.shutdown_kernel(now=False)

    kc.stop_channels()

    km.cleanup_resources()

    sys.exit(f"{message}")
    
     

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', type=str,  help='full csv filename')
    parser.add_argument('-i', '--instructions', type=str, help='markdown file for initial model prompt')
    parser.add_argument('-s', '--steps',  action='store_true', help='Indicate step by step data frame saving')
    parser.add_argument('-c', '--compact', action='store_true', help='Use compact history to save tokens, but history tracking remains')
    parser.add_argument('-l',
      '--log-level',
      type=str,
      default='WARNING',
      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
      help='Set the logging level (default: INFO)'
        )

    args = parser.parse_args()
    
    set_logger(args.log_level)

    llm_settings = LLMConfig()
    path_settings = FilePathConfig()
    ensure_directories(path_settings)
    llm = LLM(llm_settings.provider, llm_settings.llm_name, llm_settings.llm_endpoint, llm_settings.is_local) 

    system_prompt: str
    accumulated_user_queries: List[str] = []

    km = KernelManager(kernel_name='python3')
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready()

    conversation_history: List[Message] = []

    if args.datafile:
        try:
            execute_csv_load(kc, path_settings, args.datafile)
        except (TimeoutError, CSVLoadError) as e:
            logger.error("Failed to load csv file. Error:\n%s", e)
            cleanup_and_exit(kc, km)

    if args.instructions:
        query = acquire_input(kc, km, path_settings, accumulated_user_queries, args.instructions)
    else:    
        print("\nEnter in your prompt ending in _END_, or enter in !exit: \n\n")
        query = acquire_input(kc, km, path_settings, accumulated_user_queries)

    if not query:
        logger.warning("\nNo query provided.")
        cleanup_and_exit(kc, km)

    if args.steps:
        
        path_settings.output_path = str(Path(path_settings.output_path) / 'steps')
        path_settings.html_path = str(Path(path_settings.html_path) / 'steps')
        system_prompt = SYSTEM_PROMPT_WITH_STEPS.format(path=Path(path_settings.output_path).resolve())
    else:
        system_prompt = SYSTEM_PROMPT.format(path=Path(path_settings.output_path).resolve())
            
    conversation_history.append(Message(role='system', content=system_prompt))
    
    
    try:
        state = get_kernel_state(kc)
    except TimeoutError as e:
        logger.error("Timed out executing state probe while getting initial kernel state. Error:\n%s", e)
        cleanup_and_exit(kc, km)

    query = f"The current kernel state is: \n\n{state}\n\n" + query

    query_msg = Message(role='user', content=query)
    conversation_history.append(query_msg)

    
    while True:        

        logger.info(f"Accumulated prompts are:\n\n {accumulated_user_queries}")

        payload: List[Dict[str, str]] = [m.model_dump() for m in conversation_history]
        
        acceptable_code_block_counter = 0
        while True:
            # While loop will loop until acceptable code block is returned.
            result = llm.completion_call(payload)
            clean_result = strip_ansi(result['choices'][0]['message']['content'])
            logger.info(f"Code block is:\n\n{clean_result}")
            
            try:
                code_block = extract_code(clean_result)
                break
            except ValueError as e:
                logger.warning("Code block not properly delineated:\n%s\n", e)
                acceptable_code_block_counter += 1
                if acceptable_code_block_counter >= 10:
                    logger.error("Exited due to too many failures in LLM to generate acceptable code block.")
                    cleanup_and_exit(kc, km)
            
        assistant_reply = Message(role='assistant', content=clean_result)
        conversation_history.append(assistant_reply)

        try:
            res = execute_and_capture(kc, code_block)
        except TimeoutError as e:
            logger.error("Timed out executing code block. Error:\n%s", e)
            cleanup_and_exit(kc, km)

        if res['error'] is not None:  # Ignore as res will have an error filed that is default set to None unless error message is generated by jupyter_client.
            e = res['error']
            clean_trace = [strip_ansi(x) for x in e['traceback']]
            trace = [x+'\n' for x in clean_trace]

            full_error = f"""Error name: {e['ename']}\n\nError value: {e['evalue']}\n\nError Traceback:\n------\n{''.join(trace[1:])}"""

            logger.warning(full_error)

            try:
                state = get_kernel_state(kc) 
            except TimeoutError as e:
                logger.error("Timed out getting error kernel state. Error:\n%s", e)
                cleanup_and_exit(kc, km)

            logger.info(f"\n\nError KERNEL STATE IS\n\n{state}\n\n\n")
            error_message = Message(role='user',
                                    content=f"""The current kernel state is {state}\n\n. The previous code generated the following error, please fix it:\n{full_error}\n\n""")

            conversation_history.append(error_message)
            query = error_message.content
            archive_error_files(path_settings)

            continue

        
        html_files_creation_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if args.steps:
            try:
                dfs = get_all_time_sorted_files(path_settings) # Relying on LLM having saved files via prompt instructions.
            except IndexError as e:
                logger.error("Error %s:\n\nPossible no data frame was produced or folder %s is empty.", e, path_settings.top_level_output_path)
                cleanup_and_exit(kc, km)


            for i, df in enumerate(dfs):
                deliver_to_browser(df, path_settings, html_files_creation_timestamp, i, open_browser=False)
            html_steps_path = file_results(path_settings, html_files_creation_timestamp)
            for html_file in sorted(html_steps_path.iterdir()):
                if html_file.suffix == '.html':
                    logger.info(f"Opening HTML file at {html_file} in browswer.")
                    webbrowser.open(html_file.as_uri())
            
        else:
            try:
                df = get_time_sorted_file(path_settings) # Relying on LLM having saved files via prompt instructions.
            except IndexError as e:
                logger.error("Error %s:\n\nPossible no data frame was produced or folder %s is empty.", e, path_settings.output_path)
                cleanup_and_exit(kc, km)
            
            deliver_to_browser(df, path_settings, html_files_creation_timestamp)

        # Get kernel state to pass to LLM on next iteration
        try:
            state = get_kernel_state(kc)
        except TimeoutError as e:
            logger.error("Timed out getting post execution kernel state. Error:\n%s", e)
            cleanup_and_exit(kc, km)

        logger.debug("Post error-free iteration kernel state is:\n\n%s\n\n", pprint.pformat(state, indent=3))

        print("\n\nEnter in your follow-up question followed by _END_ or to quit enter in !exit: \n\n")
        query = acquire_input(kc, km, path_settings, accumulated_user_queries)
        
        if not query:  # Exit and save history
            query = f"The final kernel state is: \n\n{state}\n\nEND"
            logger.info(query)
            conversation_history.append(Message(role='user', content=query))
            save_history(conversation_history, path_settings)
            cleanup_and_exit(kc, km, f"Process complete.\nData found in {path_settings.top_level_output_path.resolve()}")

        if args.compact:
            reset_reload_context_compact_history(conversation_history, state, path_settings, query, system_prompt, accumulated_user_queries)
        else:
            query = f"The current kernel state is: \n\n{state}\n\n" + query
            conversation_history.append(Message(role='user', content=query))


if __name__ == "__main__":
    main()
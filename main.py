
import warnings
import sys
import re
import os
import time
import datetime
import json
import argparse
import queue
from typing import List, Dict, Any
from pathlib import Path
from pprint import pprint

import pandas as pd
import jupyter_client
from jupyter_client.manager import KernelManager
from jupyter_client.blocking.client import BlockingKernelClient
from rich.console import Console
from rich.markdown import Markdown

from config import LLMConfig, FilePathConfig
from llm import LLM
from message import Message
from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_STEPS, STATE_PROBE, LOAD_STATE

# Setup
warnings.filterwarnings("ignore")
jupyter_client.client.KernelClient.__del__ = lambda self: None

llm_settings = LLMConfig() #type: ignore

path_settings = FilePathConfig() #type: ignore



llm = LLM(llm_settings.provider, llm_settings.llm_api_key, llm_settings.llm_name, llm_settings.llm_endpoint, llm_settings.is_local) 


def acquire_input(manager: KernelManager, accumulated_user_queries: List[str], file_name: str | None = None) -> str | None:
    lines: List[str] = []
    
    if file_name is not None:
        path = Path(path_settings.markdown_path)
        if not path.is_dir():
            os.mkdir(path.resolve())
        with open(f'{path.resolve()} / {file_name}') as f:
            query_list = [x for x in f.readlines()]
            query = "".join(query_list)
            accumulated_user_queries.append(query)
            return query    

    while True:
        query = sys.stdin.readline()
        lines.append(query.rstrip('\n'))
        if query.lstrip()[:5] == "!exit":
            manager.shutdown_kernel()
            break
        elif query.rstrip('\n')[-5:] == "_END_":
            print(f"query is : {"".join(lines)[:-5]}\n\n")
            parsed_query = "".join(lines)[:-5]
            accumulated_user_queries.append(parsed_query)
            return parsed_query


def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)


def extract_code(text: str) -> str:
    assert '```python\n' == text[:10], "Incorrect code block start formatting."
    assert '\n```' == text[-4:], "Incorrect code block end formatting"
    
    return text[10:-4]


def get_kernel_state(client: BlockingKernelClient) -> dict[str, Any]:
    result = execute_and_capture(client, STATE_PROBE)
    for output in result['outputs']:
        if output['type'] == 'stream':
            try:
                dat = json.loads(output['text'])
                return {x:y for x,y in dat.items() if x not in {'In', 'Out', 'original_ps1', 'is_wsl'}}
            except json.JSONDecodeError:
                print("Encountered json decoding error")
    return {}


def execute_and_capture(client: BlockingKernelClient, code: str, timeout: int = 30) -> dict[str, str]:
    
    msg_id = client.execute(code)
    outputs = []
    error = None
    stream_buffers = {}
    
    while True:
        try:
            msg = client.get_iopub_msg(timeout=timeout)
        except queue.Empty:
            break
        
        msg_type = msg['msg_type']
        content = msg['content']
        
        if msg_type == 'stream':
            name = content['name']
            stream_buffers['name'] = stream_buffers.get(name, '') + content['text']
            outputs.append({'type': 'stream', 'text': content['text']})
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


def get_time_sorted_file() -> pd.DataFrame:

    #timestamp = datetime.datetime.strptime(time.ctime(), "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%d_%H-%M-%S") 
    directory = Path(path_settings.output_path).resolve() #/ timestamp
    files = sorted([f for f in directory.iterdir() if f.is_file()], key=os.path.getctime)

    # Select last which is newest        
    df = pd.read_csv(files[-1].resolve(), encoding='utf-8', index_col=0)
    return df 


def get_all_time_sorted_files() -> List[pd.DataFrame]:

    #timestamp = datetime.datetime.strptime(time.ctime(), "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%d_%H-%M-%S") 
    directory = Path(path_settings.output_path).resolve() #/ timestamp
    files = sorted([f for f in directory.iterdir() if f.is_file()], key=os.path.getctime)

    # Oldest to newest        
    dfs = [pd.read_csv(x.resolve(), encoding='utf-8', index_col=0) for x in files]
    return dfs 


def deliver_to_browser(df: pd.DataFrame, timestamp: str, counter: int | None = None):
    
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
     
    path = Path(path_settings.html_path).resolve() #/ timestamp
    
    if not path.is_dir():
        os.mkdir(path)
    print(f" Path is : {path}")

    
    if counter is None:
        path = path / f"{timestamp}.html"
        with open(path, "w") as f:
            f.write(html)
    else:
        path = path / f"{timestamp}_{counter}.html"
        with open(path, "w") as f:
            f.write(html)

    
    import webbrowser
    print(f"opening file at {path}")
    webbrowser.open(f"file://{path}")


def execute_csv_load(client: BlockingKernelClient, filename: str):
    path = Path(path_settings.input_path).resolve()
    if not path.is_dir():
        os.mkdir(path)
    x = execute_and_capture(client, LOAD_STATE.format(filepath=str(path), file=filename))
    if x['error']:
        x['error']['traceback'] = [strip_ansi(y) for y in x['error']['traceback']]
    print(x)


def save_history(conversation_history: List[Message]) -> None:
    
    timestamp = datetime.datetime.strptime(time.ctime(), "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(path_settings.history_path)
    if not path.is_dir():
        os.mkdir(path)

    with open(path / f"{timestamp}_history.txt", 'w') as f:
        f.write("\n\n".join([x.content for x in conversation_history])) 
    

def reset_reload_context_save_history(conversation_history: List[Message], state: Dict[str, Any], query: str, system_prompt: str, accumulated_user_queries: List[str]) -> None:
    # Clear and save conversation state to clean up context window 
    # conversation_history.append(Message(role='user', content=f"The ending kernel state is: \n\n{state}\n\n"))
    # save_history(conversation_history)

    conversation_history = []
    conversation_history.append(Message(role='system', content=system_prompt))

    #Test context window compression by adding in SYSTEM plus concatenation of summary of all previous user prompts (not messages as those contain code and error traces)
    
    query = f"The current kernel state is: \n\n{state}\n\n\nThis was achieved through the following user directions:\n{'\n'.join(accumulated_user_queries)}\n\nThe next user instruction is:\n\n" + query
    conversation_history.append(Message(role='user', content=query))



def main():

    system_prompt: str
    accumulated_prompts: List[str] = []

    km = KernelManager(kernel_name='python3')
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready()

    conversation_history: List[Message] = []
    print(f"Accumulated prompts:\n{accumulated_prompts}")

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', type=str,  help='full csv filename')
    parser.add_argument('-i', '--input', type=str, help='markdown file for initial model prompt')
    parser.add_argument('-s', '--steps',  action='store_true', help='Indicate step by step data frame saving')

    args = parser.parse_args()

    if args.datafile:
        execute_csv_load(kc, args.datafile)

    if args.input:
        query = acquire_input(km, accumulated_prompts, args.input)
    else:    
        print("Enter in your prompt ending in '_END_': \n\n")
        query = acquire_input(km, accumulated_prompts)

    if not query:
        sys.exit(0)

    if args.steps:
        
        path_settings.output_path = 'tmp/steps'
        path_settings.html_path = 'tmp/html_files/steps'
        system_prompt = SYSTEM_PROMPT_WITH_STEPS.format(path=Path(path_settings.output_path).resolve())
    else:
        system_prompt = SYSTEM_PROMPT.format(path=Path(path_settings.output_path).resolve())
            
    conversation_history.append(Message(role='system', content=system_prompt))
    
    # console = Console()
    # md = Markdown(query)
    # print("\n\nYou entered in:\n\n")
    # console.print(md)
    
    state = get_kernel_state(kc)
    query = f"The current kernel state is: \n\n{state}\n\n" + query

    query_msg = Message(role='user', content=query)
    conversation_history.append(query_msg)

    while True:        

        print(f"Accumulated prompts are:\n\n {accumulated_prompts}")

        payload = [m.model_dump() for m in conversation_history]
        
        while True:
            result = llm.completion_call(payload)
            clean_result = strip_ansi(result['choices'][0]['message']['content'])
            print(f"Code block is:\n\n{clean_result}")
            
            try:
                code_block = extract_code(clean_result)
                break
            except AssertionError as e:
                print(f"Failed assert:\n{e}\n")
            
        assistant_reply = Message(role='assistant', content=clean_result)
        conversation_history.append(assistant_reply)

        res = execute_and_capture(kc, code_block)

        if res['error'] is not None:
            e = res['error']
            clean_trace = [strip_ansi(x) for x in e['traceback']]
            trace = [x+'\n' for x in clean_trace]

            full_error = f"""Error name: {e['ename']}\n\nError value: {e['evalue']}\n\nError Traceback:\n------\n{''.join(trace[1:])}"""

            print(full_error)

            state = get_kernel_state(kc) 
            print("\n\nError KERNEL STATE IS\n\n")
            print(state)
            print("\n\n\n")
            error_message = Message(role='user',
                                    content=f"""The current kernel state is {state}\n\n. The previous code generated the following error, please fix it:\n{full_error}\n\n""")
            
            conversation_history.append(error_message)
            query = error_message.content
            continue

        
        html_files_creation_timestamp = datetime.datetime.strptime(time.ctime(), "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%d_%H-%M-%S")
        
        if args.steps:
            dfs = get_all_time_sorted_files()
            for dataframe in enumerate(dfs):
                deliver_to_browser(dataframe[1], html_files_creation_timestamp, dataframe[0])
            
        else:
            df = get_time_sorted_file() 
            deliver_to_browser(df, html_files_creation_timestamp)

        # Get kernel state to pass to LLM on next iteration
        state = get_kernel_state(kc)
        print("Post error-free iteration kernel state is:\n\n")
        pprint(state, indent=3)
        
        print("\n\nEnter in your follow-up question: \n\n")
        query = acquire_input(km, accumulated_prompts)
        
        if not query:  # Exit and save history
            query = f"The final kernel state is: \n\n{state}\n\nEND"
            conversation_history.append(Message(role='user', content=query))
            save_history(conversation_history)
            sys.exit(0)

        query = f"The current kernel state is: \n\n{state}\n\n" + query

        conversation_history.append(Message(role='user', content=query))
        
        #reset_reload_context_save_history(conversation_history, state, query, system_prompt, accumulated_prompts)




if __name__ == "__main__":
    main()
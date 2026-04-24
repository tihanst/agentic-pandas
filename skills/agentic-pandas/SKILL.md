---
name: pandas-agent
description: Use this skill whenever the user wants to analyze, 
  transform, filter, aggregate, or answer questions about tabular 
  data (CSV, Excel, DataFrames, etc.). The agent accepts plain English — 
  do NOT write pandas code yourself. Trigger on any data manipulation 
  request where a dataset is provided, referenced, or even if the user wants
  you to create something from scratch. The output data is saved to file 
  and opened in the user's web browser, but will return the 'head' 
  of the data frame to you to display.
---

# Pandas Agent

This skill wraps an MCP agent that generates and executes pandas code 
in a sandboxed Docker kernel. You send plain English; it handles the 
rest.

## Core workflow

1. **Start the session** with `start_session` before the first query, passing a file path if provided by the user.
2. **Send queries** with `pandas_query`. Use natural language — 
   describe what you want, not how to compute it. The agent maintains 
   state across queries, so follow-ups can reference prior results 
   ("now group that by region").
3. **End the session** with `end_session` when the user's task is 
   complete, OR when switching to an unrelated task. This shuts down 
   the Docker container. Do not skip this step.

## Rules

- Never write pandas code in your own response and ask the agent to 
  run it. Describe the goal in plain English instead.
- One active session at a time. If `start_session` reports an 
  existing session, use it rather than starting a new one.
- If `pandas_query` returns an error, call `diagnose_kernel` before 
  retrying. If the kernel is unhealthy, `end_session` then 
  `start_session` to restart cleanly.
- If API/config issues are suspected, `diagnose_environment` checks 
  what the server can see.
- `pandas_query` returns the head (first 5 rows) of the result as a 
  markdown table. Reproduce this table verbatim in your response. The 
  full result is already saved to file and opened in the user's browser. 

## Session lifecycle guarantee

Always end sessions when the user indicates that they are done, and not before. If the conversation wraps up with an active 
session, call `end_session` as part of your final response. Leaving 
containers running wastes resources. Always tell the user the file path that the results were saved to, which should
be given to you by `end_session`.

## Example

User: "What's the average order value by country in orders.csv?"

Correct: `start_session(path/to/orders.csv)` → `pandas_query("Load orders.csv and compute 
average order value grouped by country")` → present result → 
`end_session` when user indicates they are done.

Incorrect: writing `df.groupby('country')['value'].mean()` yourself 
and asking the agent to execute it.
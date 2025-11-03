# RH AI How to Run:

Switch to the `agent-loop` branch.

1. Install deps:
    ```bash
    make init
    ```
2. Start backend in a new terminal (this takes a bit of time):
    ```bash
    LFX_DEV=1 make backend
    ```
3. Start frontend in a new terminal:
    ```bash
    make frontend
    ```
You have successfully started the langflow server, with custom components enabled

To further test out Tau-Bench Airline flows, start the mcp server:
```bash
source .venv/bin/activate
python airline_tools_mcp.py --mcp-port 8000 --reload-port 8001
```

This will start the MCP server on port 8000.
When you add the server add it as: `http://localhost:8000/sse`

The reload port is to reload the database during evaluation.
You wont have to do it manually when using User Simulation Component.
You can reload by doing a GET request like follows:
```bash
curl http://localhost:8001/reload
```


---

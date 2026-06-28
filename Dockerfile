# The whole compilot application as a self-contained image: Python 3.14 +
# islpy (ISL legality engine) + clang/libomp (the OpenMP toolchain runner.py
# shells out to). Default entrypoint is the MCP server; override CMD to run the
# TUI (`compilot-tui`), the agent (`python run_agent.py ...`), or the tests.
FROM python:3.14-slim

# clang + libomp-dev: runner.py compiles generated kernels with `clang -fopenmp`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

# stdio MCP server — run with `docker run -i ghcr.io/cluster2600/cluster_compilot`.
ENTRYPOINT ["compilot-mcp"]

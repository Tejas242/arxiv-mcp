FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY main.py ./
COPY README.md ./

RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Create non-root user for security
RUN groupadd -r arxiv && useradd -r -g arxiv arxiv

WORKDIR /app

# Copy built application from builder stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/main.py /app/main.py
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Make virtual environment accessible
ENV PATH="/app/.venv/bin:$PATH"

# Change ownership to non-root user
RUN chown -R arxiv:arxiv /app
USER arxiv

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose port (MCP servers typically run on stdio, but adding for potential HTTP mode)
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "main.py"]

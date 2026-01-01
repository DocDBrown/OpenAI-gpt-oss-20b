# Dockerfile
# syntax=docker/dockerfile:1

ARG BIN_NAME=gpt-oss-20b

# 1) Rust client build stage (kept as-is; not built when target=llama)
FROM rust:1.92.0-bookworm AS client-builder
WORKDIR /app

ARG BIN_NAME=gpt-oss-20b

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN set -eux; \
    cargo build --locked --release; \
    test -f "target/release/${BIN_NAME}"; \
    strip "target/release/${BIN_NAME}" || true

# 2) Rust client runtime (distroless) (kept as-is; not built when target=llama)
FROM gcr.io/distroless/cc-debian12:nonroot AS client
WORKDIR /app

ARG BIN_NAME=gpt-oss-20b
COPY --from=client-builder /app/target/release/${BIN_NAME} /app/client

ENTRYPOINT ["/app/client"]

# 3) Build llama-server from source for ARM64
FROM debian:bookworm AS llama-builder
WORKDIR /src

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      build-essential \
      cmake \
      ninja-build; \
    rm -rf /var/lib/apt/lists/*

ARG LLAMA_CPP_REF=master

RUN set -eux; \
    git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp.git .; \
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release; \
    cmake --build build --config Release --target llama-server

RUN set -eux; \
    mkdir -p /out; \
    if [ -x build/bin/llama-server ]; then \
      cp build/bin/llama-server /out/llama-server; \
    elif [ -x build/llama-server ]; then \
      cp build/llama-server /out/llama-server; \
    else \
      echo "ERROR: llama-server not found after build" >&2; \
      find build -maxdepth 4 -type f -name 'llama-server' -print >&2 || true; \
      exit 1; \
    fi; \
    strip /out/llama-server || true

# 4) Runtime image (ARM64) with llama-server binary and model baked in
FROM debian:bookworm-slim AS llama
WORKDIR /models
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates libgomp1; \
    rm -rf /var/lib/apt/lists/*

COPY --from=llama-builder /out/llama-server /usr/local/bin/llama-server
COPY models/gpt-oss-20b-Q5_K_M.gguf /models/gpt-oss-20b-Q5_K_M.gguf

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/llama-server"]

CMD ["-m", "/models/gpt-oss-20b-Q5_K_M.gguf", "--host", "0.0.0.0", "--port", "8080", "-c", "0", "-n", "512", "--jinja", "--n-gpu-layers", "0"]

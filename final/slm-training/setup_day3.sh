#!/bin/bash
# Day 3 setup: run this ONCE, in a terminal, at the start of the day. It takes 10 to 20 minutes and can run while you listen.
# Everything is written to /workspace/day3_setup.log. The last line says SETUP DONE when it has finished.
cd /workspace
(
  echo "1/3 llama.cpp (for Lab 18)"
  [ -d /workspace/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
  echo "2/3 GGUF tools (for Lab 18)"
  pip install --no-cache-dir -q gguf sentencepiece llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
  echo "3/3 vLLM, in its own environment (for Lab 19)"
  [ -d /root/vllm-venv ] || python -m venv /root/vllm-venv
  /root/vllm-venv/bin/pip install --no-cache-dir -q vllm
  echo "SETUP DONE"
) > /workspace/day3_setup.log 2>&1 &
echo "started: follow it with   tail -f /workspace/day3_setup.log"

# SLM training: all labs in one folder

This folder holds **everything for all three days**: the notebooks of Day 1, Day 2 and Day 3 (and four optional extra labs), the helper files they import, the small data files and the install list. You do **not** need any other folder or zip.

## What is inside (`slm-training/`)

| Item | What it is |
|---|---|
| `day1/` | Labs 1 to 8: tokens, transformers, evaluation, RAG, guardrails |
| `day2/` | Labs 9 to 15: tools and ReAct, LangGraph, memory, guardrails for agents, evals, the dialogue manager |
| `day3/` | Labs 16 to 21: prompt engineering, fine-tuning, real quantisation, serving many users, when the intent is wrong, the finale |
| `day3_extra/` | four **optional** extra labs (X1 to X4) for when you have time: advanced prompting, multi-agent with LangGraph for cost and latency, a TV dialogue manager, a voice assistant. Do them after Lab 21 |
| `*.py` helper files | code the notebooks import (plain Python: open them if you are curious) |
| `data/` | the (fictional) company, appliance and TV data; the 72 test messages; spoken clips for the voice labs; two prepared adapters used only as a fallback if your own training fails |
| `requirements.txt` | the packages to install |
| `setup_day3.sh` | one-time setup for Labs 18 and 19 (needed on Day 3 only) |

## Run it on Runpod

1. Create a Pod: an NVIDIA GPU, template **Runpod PyTorch 2.8.0 (CUDA 12.8.1)**, a **network volume of at least 30 GB** at `/workspace`, container disk 50 GB. Days 1 and 2 run on 16 GB; **Day 3 needs 24 GB** (an RTX 4090 is what we tested).
2. Open Jupyter Lab and upload `SLM-Training-All-Days.zip` into `/workspace`.
3. Open a terminal and run:
   ```bash
   cd /workspace
   unzip -o SLM-Training-All-Days.zip
   cd SLM-Training-All-Days/slm-training
   pip install --no-cache-dir -r requirements.txt
   ```
   This takes a few minutes. Do not reinstall PyTorch: the template has it.
4. **Day 3 only:** start the one-time setup first; it runs in the background for 10 to 20 minutes:
   ```bash
   bash setup_day3.sh
   ```
   Follow it with `tail -f /workspace/day3_setup.log`; it ends with `SETUP DONE`. Do not wait for it: start Lab 16.
5. Open the first notebook of the day and work through the labs **in number order**. In each notebook use **Kernel, Restart Kernel and Run All Cells**.
6. Models are downloaded the first time they are used and kept on the volume: Qwen2.5-1.5B (about 3 GB), Qwen2.5-3B (about 6 GB), and on Day 3 Whisper small and a small text-to-speech model (about 1 GB).
7. **Stop the Pod** when you finish (Pods, the three dots, Stop). Never Terminate.

After you stop and start a Pod, the installed packages and the Day 3 setup are erased: run `pip install --no-cache-dir -r requirements.txt` again, and `bash setup_day3.sh` again on Day 3. Your files and the downloaded models stay on the volume.

## Day 1: Transformers, RAG and guardrails

| Notebook | Lab | Time |
|---|---|---|
| `day1/lab01_setup.ipynb` | Lab 1 : Setup: your GPU workbench | 15 min |
| `day1/lab02_tokens.ipynb` | Lab 2 : Tokens and chat templates | 20 min |
| `day1/lab03_transformer_vs_lstm.ipynb` | Lab 3 : Why Transformers beat RNNs and LSTMs | 40 min |
| `day1/lab04_inside.ipynb` | Lab 4 : Inside the model: probabilities, perplexity and attention | 30 min |
| `day1/lab05_evaluation.ipynb` | Lab 5 : Measure first: an evaluation set and the hallucination baseline | 25 min |
| `day1/lab06_rag.ipynb` | Lab 6 : RAG: ground the answers in the policy | 30 min |
| `day1/lab07_rag_experiments.ipynb` | Lab 7 : RAG experiments: change one thing, watch what breaks | 30 min |
| `day1/lab08_guardrails.ipynb` | Lab 8 : Guardrails: rules before and after the model | 40 min |

## Day 2: Agents, LangGraph and the dialogue manager

| Notebook | Lab | Time |
|---|---|---|
| `day2/lab09_tools_and_react.ipynb` | Lab 9 : Tools and ReAct: build an agent that acts, with the small model | 60 min |
| `day2/lab10_langgraph_basics.ipynb` | Lab 10 : LangGraph basics: state, nodes and edges, with the small model from the first line | 40 min |
| `day2/lab11_routing_loops_fanout.ipynb` | Lab 11 : Routing, loops, reducers, Command and fan-out | 50 min |
| `day2/lab12_memory_and_human_in_the_loop.ipynb` | Lab 12 : Memory and human in the loop | 55 min |
| `day2/lab13_guardrails_for_agents.ipynb` | Lab 13 : Guardrails for an agent, with NVIDIA NeMo Guardrails | 55 min |
| `day2/lab14_evals_with_deepeval.ipynb` | Lab 14 : Evals for your agent: a golden dataset and DeepEval | 40 min |
| `day2/lab15_dialogue_manager.ipynb` | Lab 15 : The dialogue manager: the model understands, the graph controls | 45 min |

## Day 3: Optimise a small model for the edge

| Notebook | Lab | Time |
|---|---|---|
| `day3/lab16_prompt_engineering.ipynb` | Lab 16 : Optimisation 1: prompt engineering, the cheapest change | 40 min |
| `day3/lab17_finetune_intent.ipynb` | Lab 17 : Optimisation 2: teach the model, on a small memory budget | 60 min |
| `day3/lab18_quantise_for_real.ipynb` | Lab 18 : Optimisation 3: make the model small, for real | 55 min |
| `day3/lab19_serving_many_users.ipynb` | Lab 19 : Optimisation 4: many people, one device | 75 min |
| `day3/lab20_when_intent_is_wrong.ipynb` | Lab 20 : When the intent is wrong | 45 min |
| `day3/lab21_check_the_brief.ipynb` | Lab 21 : The finale: check the brief, then hear it work | 25 min |

### Optional extra labs (`day3_extra/`), in this order, only if you have spare time

| Notebook | Lab | Time |
|---|---|---|
| `day3_extra/extra_x1_advanced_prompting.ipynb` | Extra lab X1 : Advanced prompt engineering, measured | 55 min |
| `day3_extra/extra_x2_multi_agent_cost_latency.ipynb` | Extra X2 : Several agents in one graph, designed for cost and latency | 55 min |
| `day3_extra/extra_x3_tv_dialogue_manager.ipynb` | Extra Lab X3 : A complete dialogue manager for a TV, with the small model | 60 min |
| `day3_extra/extra_x4_voice_assistant.ipynb` | Extra X4 : An end-to-end voice assistant |  |

Extra lab X1 needs the GGUF files that Lab 18 makes. X3 trains a small TV adapter (about 2 minutes) and X4 uses it.

## If something is slow or stuck

- Do **not** wait more than a few minutes on any cell. Interrupt the kernel (Kernel, Interrupt) and ask the trainer.
- If your Lab 17 training fails or is too slow, go on to Lab 18: it uses the prepared adapter when yours is missing. The same holds for the TV adapter in extra lab X3.
- A first message on a new `.gguf` file can take 10 seconds or more while the file is read from disk: this is normal, once per file.
- If the volume is full ("Disk quota exceeded"), delete old model folders you no longer need in `/workspace/.cache/huggingface/hub`, and always install with `--no-cache-dir`.
- Keep your own work in copies: save a notebook as `labNN_assignment.ipynb` before you change it, so the original stays as it was.

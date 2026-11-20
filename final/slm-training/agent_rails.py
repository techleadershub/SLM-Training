"""NVIDIA NeMo Guardrails for a LangChain agent, with the course's small model as the judge (used by Lab 13).

    from agent_rails import write_rails_config, GuardrailsMiddleware
    folder = write_rails_config(input_policy, output_policy)          # writes config.yml and rails.co
    agent = create_agent(model=..., tools=[...], middleware=[GuardrailsMiddleware(config_path=folder)])

The rails are the built-in `self check input` and `self check output`: NeMo Guardrails asks a language model
"should this be blocked? Yes or No". The model is the small model running on this machine (`graph_slm.slm`), so nothing leaves it.
`GuardrailsMiddleware` is NVIDIA's own integration with LangChain agents: it runs the input rails before the model and the
output rails after it.
"""
import os
import tempfile

import nest_asyncio

nest_asyncio.apply()                      # lets NeMo Guardrails run inside a notebook

from nemoguardrails import LLMResponse, UsageInfo, register_provider  # noqa: E402
from nemoguardrails.integrations.langchain.middleware import GuardrailsMiddleware  # noqa: E402,F401

from graph_slm import slm  # noqa: E402


class LocalSLMModel:
    """The adapter NeMo Guardrails needs: it asks a question, we ask the small model and return the text."""

    def __init__(self, model, **kwargs):
        self._model = model

    model_name = property(lambda self: self._model)
    provider_name = property(lambda self: "local_slm")
    provider_url = property(lambda self: None)

    async def generate_async(self, prompt, *, stop=None, **kwargs):
        chat = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else \
            [{"role": getattr(m, "role", "user"), "content": m.content} for m in prompt]
        text = slm(chat, max_new_tokens=int(kwargs.get("max_tokens") or 8))
        return LLMResponse(content=text, model=self._model, finish_reason="stop", usage=UsageInfo(input_tokens=0, output_tokens=len(text)))

    async def stream_async(self, prompt, *, stop=None, **kwargs):
        raise NotImplementedError


register_provider("local_slm", LocalSLMModel)
COLANG = 'define bot refuse to respond\n  "I\'m sorry, I can\'t help with that."\n'


def _shots(examples, question):
    return "\n\n".join(f'Text: "{t}"\n{question} {a}' for t, a in examples)


def write_rails_config(input_policy, output_policy, in_examples=(), out_examples=(),
                       in_question="Should this message be blocked?", out_question="Should this reply be blocked?", folder=None):
    """Write a NeMo Guardrails configuration folder (config.yml + rails.co) and return its path."""
    ind = lambda t: "\n".join("      " + line for line in t.splitlines())     # noqa: E731
    inp = input_policy + "\n" + (f"Examples:\n{_shots(in_examples, in_question)}\n\n" if in_examples else "") + \
        f'Text: "{{{{ user_input }}}}"\n{in_question} Answer with one word, Yes or No.'
    out = output_policy + "\n" + (f"Examples:\n{_shots(out_examples, out_question)}\n\n" if out_examples else "") + \
        f'Text: "{{{{ bot_response }}}}"\n{out_question} Answer with one word, Yes or No.'
    yaml = ("models:\n  - type: main\n    engine: local_slm\n    model: qwen\n"
            "rails:\n  input:\n    flows:\n      - self check input\n  output:\n    flows:\n      - self check output\n"
            "prompts:\n  - task: self_check_input\n    content: |-\n" + ind(inp) +
            "\n  - task: self_check_output\n    content: |-\n" + ind(out) + "\n")
    folder = folder or tempfile.mkdtemp(prefix="agent_rails_")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.yml"), "w", encoding="utf-8") as f:
        f.write(yaml)
    with open(os.path.join(folder, "rails.co"), "w", encoding="utf-8") as f:
        f.write(COLANG)
    return folder

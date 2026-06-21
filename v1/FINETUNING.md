# Side note — Fine-tuning the explanation LLM (reference, not implemented)

The system works **without** fine-tuning: the LLM only rephrases facts the query engine already
computed, so prompt engineering is usually enough. Fine-tune only for a consistent house style,
domain vocabulary, or to make a tiny model reliable with a short prompt.

## Ladder (easiest → hardest)

1. **Prompt + few-shot (no training)** — put 2–3 (facts → ideal explanation) examples in the system
   prompt. Start here.
2. **Ollama `Modelfile`** — bake the system prompt + params into a named model (no GPU training):
   ```dockerfile
   FROM llama3.2:3b
   SYSTEM "You explain pre-computed sensor statistics. Never change, round, or invent numbers."
   PARAMETER temperature 0.2
   ```
   `ollama create tshci-explainer -f Modelfile` → use `model="tshci-explainer"`.
3. **QLoRA fine-tuning (real training)**:
   - **Data:** ~300–2,000 JSONL pairs of `computed facts → ideal explanation`. Generate cheaply by
     running the query engine over many questions and writing/auto-generating reference answers.
   - **Base:** small instruct model (Llama-3.2-3B / Qwen2.5-3B), 4-bit QLoRA.
   - **Tools:** Unsloth (easiest, low VRAM), HF PEFT+TRL `SFTTrainer`, or axolotl.
   - **Hardware:** RTX 5070 (8 GB) handles QLoRA on 1–3B comfortably (7B is tight).
   - **Export:** train LoRA → merge/convert to GGUF (llama.cpp) → import into Ollama via Modelfile.
   - **Critical eval:** parse numbers out of the model's output and compare to the inputs — they
     **must match exactly**. This reuses the project's trust principle to validate the fine-tune.

## Recommendation for the MSc
Use Level 1 or 2 for submission; mention QLoRA as future work. Key point for the report: the
architecture is robust to a weak LLM because numbers come from code, not the model — so fine-tuning
is an optional polish, not a correctness requirement.

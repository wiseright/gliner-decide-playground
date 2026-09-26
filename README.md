# Your LLM Is Overqualified for This Job

**Fast, local decisions with GLiNER2.5-Decide — from ticket routing to a Space Invaders copilot you can talk to.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: GLiNER2.5--Decide](https://img.shields.io/badge/%F0%9F%A4%97%20model-GLiNER2.5--Decide-ffcc00)](https://huggingface.co/fastino/GLiNER2.5-Decide)
[![Runs on a free Colab T4](https://img.shields.io/badge/runs%20on-free%20Colab%20T4-f9ab00)](https://colab.research.google.com/)

Companion repository for the Medium article **Your LLM Is Overqualified for This Job**
_(link added here as soon as it is published)._

---

A lot of what we send to an LLM is a multiple-choice question in disguise.
[GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) is a 340M
encoder that answers those by **choosing**, not by writing: you give it text and
a set of candidate labels, it scores them all in a single forward pass and
returns the winner. No generation, no JSON that breaks at 2 a.m., no label that
wasn't on your list.

The whole API surface is this:

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")

model.classify_text(
    "Battery dies before lunch, but the screen is gorgeous.",
    {"sentiment": ["positive", "negative", "mixed", "neutral"]},
)
# -> {"sentiment": "mixed"}
```

This repo puts that idea through three escalating tests: **six things worth
building**, a **like-for-like latency benchmark against quantised LLMs**, and a
**real-time game loop** where being slow means losing.

---

## What's in here

| | File | What it shows |
| --- | --- | --- |
| §2 | [`02_gliner_decide_applications.ipynb`](02_gliner_decide_applications.ipynb) | Six applications, a dozen lines each: LLM router, prompt-injection gate, agent tool-call guardrail, offline smart-home commands, GitHub issue triage, live-chat moderation throughput. |
| §3 | [`03_latency_benchmark_gliner_vs_qwen.ipynb`](03_latency_benchmark_gliner_vs_qwen.ipynb) | End-to-end latency and accuracy vs 4-bit Qwen, same inputs, same output contract, same validator, same GPU. |
| §4 | [`04_space_invaders/`](04_space_invaders/) | v0.1 — the model reads the game state, code decides. v0.2 — a copilot you steer by voice or plain English. |

Section numbers match the article; there is no `01_`.

Run the notebooks on a free Colab T4 (`Runtime → Change runtime type → T4 GPU`):

[![Open §2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiseright/gliner-decide-playground/blob/main/02_gliner_decide_applications.ipynb) §2 — applications

[![Open §3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiseright/gliner-decide-playground/blob/main/03_latency_benchmark_gliner_vs_qwen.ipynb) §3 — benchmark

---

## Quickstart

```bash
git clone https://github.com/wiseright/gliner-decide-playground.git
cd gliner-decide-playground

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is grouped by section, so you can install only the part you
want. The notebooks `pip install` what they need inline, so on Colab you can skip
this step entirely.

> **Heads-up on `gliner2` 2.0.0:** it imports `transformers`, `peft` and
> `accelerate` at load time but does not declare them as dependencies, so
> `pip install gliner2` on its own fails with `ModuleNotFoundError`. On Colab it
> works by accident, because those packages are pre-installed. Install
> `gliner2 transformers peft accelerate` together.
>
> **No NVIDIA GPU?** Install the CPU wheel explicitly, or pip will pull ~3 GB of
> CUDA libraries you cannot use:
> `pip install --index-url https://download.pytorch.org/whl/cpu torch`

The Space Invaders scripts need a few more moving parts; see
[`04_space_invaders/README.md`](04_space_invaders/README.md).

---

## The numbers (§3)

End-to-end decision latency on a free Colab T4, batch size 1, 20 hand-labelled
tasks × 3 runs = 60 timed calls per model. "End-to-end" means input text →
parsed, validated JSON, prompt formatting and parsing included.

| Model | Size | p50 | p95 | Valid JSON | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| **GLiNER2.5-Decide** | 340M, fp32 | **55 ms** | **77 ms** | 100% | 70% |
| Qwen3-4B-Instruct-2507 | 4B, 4-bit NF4 | 746 ms | 1,455 ms | 100% | **90%** |
| Qwen3.5-2B | 2B, 4-bit NF4 | 897 ms | 1,492 ms | 95% | 70% |

**Latency:** 13.5× and 16.2× faster on the median. The gap widens in the p95,
the slow tail anything waiting on the model actually feels: GLiNER's worst 5%
stays under 80 ms while both LLMs go past a second and a half.

**Accuracy:** the 4B model wins, clearly — 18/20 against 14/20 — with ~12× the
parameters. The size-fair comparison is the 2B, which lands on exactly GLiNER's
70% while being 16× slower and breaking the output format in 5% of calls. And
GLiNER ran here with plain label names: no descriptions, no fine-tuning, both of
which it supports.

**Two honest caveats.** Twenty hand-labelled examples are a sanity check, not a
leaderboard — each task is worth 5 points, so two answers swing the ranking. And
the 2B being slower than the 4B is not "the effect of size": they are two
different architectures on an older GPU, where implementation maturity matters
as much as parameter count.

> The notebook benchmarks **one** LLM per run. Set `QWEN_ID` and re-run the Qwen
> section to reproduce the other row; results are appended to the same table.

---

## Space Invaders (§4)

The strictest latency budget in the repo. The Atari 2600 runs at 60 fps, and
with a frameskip of 3 a new decision is needed **every 50 ms**.

> Why frameskip 3 and not the default 4? Space Invaders draws bullets on
> alternate frames. With an even frameskip you always land on the same kind of
> frame and the enemy missiles vanish — from the video, from the state, from
> everything. The first version played blind for exactly this reason.

**v0.1 — [`space_invaders_gliner.py`](04_space_invaders/space_invaders_gliner.py).**
The first attempt asked the model to choose the move: *"if there is danger, which
way should the ship go, otherwise…"*. It failed badly — that question is a chain
of reasoning, and the model card says plainly that it does not reason. It
matched the safest-looking label, `stay`, over and over.

Splitting the work fixed it. The model answers only questions whose answer is
sitting right there in the text; the strategy lives in five lines of code:

```python
PERCEPTION = {
    "missile": {"labels": ["left", "right", "above", "none"],
                "prompt": "Where is the enemy missile relative to the ship?"},
    "alien":   {"labels": ["left", "right", "above", "none"],
                "prompt": "Where is the nearest alien relative to the ship?"},
    "gun":     {"labels": ["ready", "reloading"],
                "prompt": "Is the gun ready or reloading?"},
}
```

▶ [`assets/space_invaders_gliner_perception.mp4`](assets/space_invaders_gliner_perception.mp4) — v0.1 playing, perception schema.

It works, and it proves the model fits inside a 50 ms loop. It also proves
something less flattering: the code reads exact coordinates out of the Atari
RAM, writes them into a sentence, and the model reads the same facts back out.
A round trip. A handful of `if`/`else` lines does it faster and never misreads.

**If your input is already structured, you don't need a language model to read
it. Decision models earn their place on language you don't control.**

**v0.2 — [`space_invaders_copilot.py`](04_space_invaders/space_invaders_copilot.py).**
So the language comes from a person. The rule-based player drives, but now it
has knobs — play style, screen zone, target priority — and a human turns them by
speaking:

```python
model.classify_text("Stay behind the left bunker and don't take risks", COMMAND_SCHEMA)
# -> {"mode": "defensive", "zone": "left", "focus": "nearest_alien"}
```

This is what `if`/`else` is bad at. "Play it safe", "don't take risks" and
"careful, hide behind the shields" mean the same thing; "don't go right"
contains the word *right* and means the opposite; one sentence can change three
settings at once.

The model is **not** called every frame — only when an order arrives. It's
cruise control, not a steering wheel. Latency still matters, because the game
doesn't pause while the model thinks: at 55 ms the ship reacts within one game
step, at Qwen's 746 ms it follows the old order for 15 more steps, and at its
p95 for almost 30. That's the difference between a copilot and a copilot who
replies by letter.

---

## Using it in your own project

Three things worth knowing before you ship one of these:

1. **The label list *is* the product.** Change the list, get a new classifier,
   train nothing. Labels can carry a short description — that's where you encode
   your policy instead of hoping the label name is self-explanatory.
2. **Always add an `other` / `unknown` label.** The model can only pick from your
   list, so if the right answer isn't there it will confidently choose the least
   wrong option.
3. **"Never invents a label" is not "never wrong."** Guaranteed-valid output is a
   real and useful property — you will never get a broken JSON or a category that
   doesn't exist. But the failure mode just moves: instead of inventing something,
   the model confidently ticks the wrong valid box. Test it on your own data.

---

## ⚠️ Disclaimer

* **Not affiliated with anyone.** This is an independent, personal project. It
  is not affiliated with, endorsed by or sponsored by Fastino, TypeSafe AI,
  Alibaba / Qwen, Hugging Face, the Farama Foundation, Atari or Taito. All
  trademarks belong to their respective owners.
* **The benchmark is a sanity check, not a leaderboard.** One Colab T4, one
  session, 20 hand-labelled examples, batch size 1, no tuning on either side.
  Hardware, driver and library versions move these numbers. Reproduce them
  before quoting them.
* **Third-party claims are quoted, not verified.** Vendor benchmark figures
  mentioned in the article and here come from the vendors' own published
  material and are labelled as such.
* **The security examples are demonstrations, not security controls.** The
  prompt-injection gate and the agent tool-call guardrail exist to show what a
  decision model can do in a dozen lines. A classifier is one layer of defence
  with a real false-negative rate; do not put either in front of anything that
  matters without your own evaluation, threat model and independent controls.
* **No Atari ROMs here.** This repository contains no ROM and no ROM download
  link. ROM provisioning is handled by `ale-py`; follow its documentation and
  applicable law. The gameplay clip in `assets/` is a short recording of
  emulated play, included to illustrate a technical point.
* **No warranty.** Provided "as is", as stated in [`LICENSE`](LICENSE). Model
  outputs can be wrong. You are responsible for whatever you build on them.

---

## Licence and attribution

The code in this repository is released under the **[MIT License](LICENSE)**.

That covers this repository's own code only. Models, libraries, the Atari
emulator and any game content keep their own licences and terms — all of them
are listed, with sources, in **[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)**.
Two worth flagging up front: `ale-py` is **GPL-2.0** (imported as a runtime
dependency, never bundled here), and Qwen checkpoints carry model-card terms
that differ between releases.

### Citing

If this repository was useful, please cite the underlying work first — the
research is theirs, the glue code is mine:

```bibtex
@article{gliner2,
  title  = {GLiNER2: An Efficient Multi-Task Information Extraction System
            with Schema-Driven Interface},
  journal = {arXiv preprint arXiv:2507.18546},
  year   = {2025},
  url    = {https://arxiv.org/abs/2507.18546}
}
```

For the repository itself, GitHub's *Cite this repository* button reads
[`CITATION.cff`](CITATION.cff).

---

## Contributing

Issues and PRs are welcome — especially reproductions of the benchmark on other
hardware, which is the number most in need of independent data. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

Built something fun with it? I'd love to hear about it in the article's comments.

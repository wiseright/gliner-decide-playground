# Third-party notices

The MIT licence in [`LICENSE`](LICENSE) covers **the code written for this
repository** and nothing else. This repository does not vendor, redistribute or
mirror any third-party model weights, datasets or ROMs: everything below is
downloaded by you, at run time, from its own source and under its own terms.

Licences are listed as a convenience, checked on **2026-09-26**. They can
change. Before you rely on any of them, read the licence at the linked source.

## Models

| Component | Source | Licence |
| --- | --- | --- |
| **GLiNER2.5-Decide** (340M) | [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) | Apache-2.0, per the model card |
| **Qwen3-4B-Instruct-2507** | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | See the model card |
| **Qwen3.5-2B** | Hugging Face model card | See the model card |
| **Whisper** (`base.en`, optional, via faster-whisper) | [openai/whisper](https://github.com/openai/whisper) | MIT |

The Qwen rows are deliberately vague: model-card terms differ between Qwen
releases and have changed over time. Check the card for the exact checkpoint
you load before using it for anything beyond a local experiment.

## Libraries

| Library | Used for | Licence |
| --- | --- | --- |
| [gliner2](https://github.com/fastino-ai/GLiNER2) | the decision model API | See the project page |
| [transformers](https://github.com/huggingface/transformers) | loading Qwen | Apache-2.0 |
| [accelerate](https://github.com/huggingface/accelerate) | device placement | Apache-2.0 |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 4-bit NF4 quantisation | MIT |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | the environment API | MIT |
| [ale-py / Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | the Atari emulator | **GPL-2.0** |
| [OCAtari](https://github.com/k4ntz/OC_Atari) | objects from Atari RAM | See the project page |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | speech-to-text (optional) | MIT |
| [imageio](https://github.com/imageio/imageio) / imageio-ffmpeg | video export | BSD-2-Clause |

**A note on `ale-py` (GPL-2.0).** The Arcade Learning Environment is GPLv2. This
repository does not include, link against or distribute any part of it: the
scripts in `04_space_invaders/` import it as an ordinary runtime dependency that
you install yourself. If you redistribute a *combined* work that bundles ALE, the
GPL's terms apply to that combination — not to this repository's MIT-licensed
source.

## Atari ROMs and game content

* This repository contains **no Atari ROM**, no ROM download link, and no
  instructions for obtaining one. ROM provisioning is handled entirely by
  `ale-py`; follow its documentation and make sure your use complies with the
  law where you are.
* **Space Invaders** is a trademark of Taito Corporation. The Atari 2600
  version was published by Atari, Inc. Neither company is affiliated with,
  endorses or sponsors this project.
* `assets/space_invaders_gliner_perception.mp4` is a short screen recording of
  emulated gameplay, included to illustrate the article's technical argument.
  The MIT licence applies to this repository's own code; it does **not** grant
  any right in the underlying game, its audiovisual content or its trademarks,
  which remain with their respective owners. It is reproduced here as brief,
  non-commercial, illustrative use in the context of technical commentary.

## Prior work referenced

* **GLiNER2** — Fastino AI. Paper: [arXiv:2507.18546](https://arxiv.org/abs/2507.18546).
* **GLiNER** — Zaratiana et al., *GLiNER: Generalist Model for Named Entity
  Recognition using Bidirectional Transformer*, [arXiv:2311.08526](https://arxiv.org/abs/2311.08526).
* **DeBERTa-v3** — He et al., [arXiv:2111.09543](https://arxiv.org/abs/2111.09543). The
  encoder underneath GLiNER2.5-Decide.
* **Jev** — TypeSafe AI. Mentioned in the article for context only. It is a
  closed, API-only product; nothing in this repository calls it, reproduces it,
  or benchmarks it. Any figure quoted about it comes from the vendor's own
  published material.
* **Thinking, Fast and Slow** — Daniel Kahneman, 2011. The "System 1 / System 2"
  framing is cited for the idea only.

If you are a rights holder and something here needs correcting or removing,
please open an issue and it will be handled promptly.

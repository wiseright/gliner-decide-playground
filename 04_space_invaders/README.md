# §4 — Space Invaders

Two experiments that put GLiNER2.5-Decide inside a 50 ms real-time loop, and the
lesson that came out of them.

| Script | Version | What it does |
| --- | --- | --- |
| [`space_invaders_gliner.py`](space_invaders_gliner.py) | v0.1 | The model reads the game state; plain code decides the move. Includes the rule-based baseline and the failed "let the model choose the move" schema. |
| [`space_invaders_copilot.py`](space_invaders_copilot.py) | v0.2 | The rule-based player drives; a human copilot changes its strategy by typing or speaking. |

## Install

```bash
pip install gliner2 "gymnasium[atari]" ale-py ocatari imageio imageio-ffmpeg
pip install faster-whisper sounddevice      # only for --input voice
```

ROMs are provisioned by `ale-py` — this repository ships none. See
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

A GPU helps but is not required; both scripts fall back to CPU automatically.

## v0.1 — perception vs policy

```bash
# rule-based baseline: the reference score
python space_invaders_gliner.py --policy rules --render video

# the model reads the state, code decides (this one works)
python space_invaders_gliner.py --policy gliner --schema perception --render video --verbose

# the model picks the move itself (this one doesn't: it asks the model to reason)
python space_invaders_gliner.py --policy gliner --schema policy --render video --verbose
```

With `--policy gliner` the script also compares every decision against the
rule-based baseline on the same state, and prints the agreement rate, the most
frequent disagreements, and the decision latency against the frame budget.

| Flag | Default | Notes |
| --- | --- | --- |
| `--policy` | `gliner` | `gliner` or `rules` |
| `--schema` | `perception` | `perception` (read + decide in code) or `policy` (model decides) |
| `--render` | `none` | `human`, `video` (writes an mp4), or `none` |
| `--frameskip` | `3` | See the note below — don't set it to an even number |
| `--max-steps` | `3000` | |
| `--verbose` | off | One line per step: action + the sentence the model saw |

## v0.2 — the copilot

```bash
# local, keyboard
python space_invaders_copilot.py --render human

# local, push-to-talk with Whisper
python space_invaders_copilot.py --render human --input voice

# Colab or a reproducible demo: orders scheduled by game step
python space_invaders_copilot.py --render video --input script \
    --script "100:1; 500:2; 900:Stay on the left and don't take risks" --max-steps 1500
```

Three ways to give an order:

* **Shortcuts** — `1`, `2`, `3` send a preset phrase. The phrase still goes
  through the model; the number just saves typing.
* **Free text** — press `4`, or simply type any English sentence.
* **Voice** — `--input voice`: press Enter and speak for three seconds.
  faster-whisper transcribes, GLiNER decides.

Each `--script` entry is `step:order`. With frameskip 3 a step lasts 50 ms, so
step 100 is about five seconds into the game.

| Flag | Default | Notes |
| --- | --- | --- |
| `--input` | `keyboard` | `keyboard`, `voice`, or `script` |
| `--script` | — | `"150:1; 600:Stay left and play safe"` |
| `--render` | `human` | `human`, `video`, or `none` |
| `--frameskip` | `3` | |
| `--max-steps` | `5000` | |
| `--voice-seconds` | `3.0` | Recording window per push-to-talk |

The model runs **only** when an order arrives, and it runs in a worker thread:
orders go in on one queue, interpreted strategies come out on another, and the
game loop only ever drains the second one. So the loop never blocks on
inference — on a slow CPU the ship reacts a few steps later rather than the
whole game stuttering. The one exception is `--input script`, which interprets
inline on purpose so that a scheduled demo stays reproducible step by step (and
in script mode the game isn't running at real speed anyway).

The script reports the interpretation latency against the length of one game step.

## Why frameskip 3

Space Invaders draws bullets on alternate frames, a trick the original hardware
used to save resources. With an even frameskip you always sample the same kind
of frame and the enemy missiles disappear from the extracted state entirely.
Old versions of Gym used 3 for this game for the same reason.

## How the text pipeline works

GLiNER reads text, not pixels, so there is a small translator between the game
and the model:

1. **Run the game** with Gymnasium + the Arcade Learning Environment.
   `env.step()` is the control API: `LEFT`, `RIGHT`, `FIRE`, `NOOP`.
2. **Read the state** with OCAtari, which extracts objects from the Atari's RAM
   — player, aliens, bullets, each with coordinates. No computer vision needed.
3. **Turn the state into a sentence**, e.g. *"DANGER: an enemy missile is falling
   to the LEFT of the ship. The nearest alien is to the RIGHT of the ship. The
   gun is ready."*
4. **Ask the model**, then map the answer back to an ALE action.

Step 3 is exactly why v0.1 is an interesting failure rather than a useful
product: it serialises structured data into language so a language model can
parse it back. v0.2 removes the round trip by pointing the model at input that
was language to begin with — a person talking.

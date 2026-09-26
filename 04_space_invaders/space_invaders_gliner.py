"""
Space Invaders, v0.1 - GLiNER2.5-Decide in a real-time game loop.

GLiNER reads text, not pixels. So we read the game objects straight from the
Atari RAM with OCAtari, turn them into a sentence, and ask the model.

Two question schemas:
  --schema perception  (default) the model READS the situation with direct questions
                       (which side is the missile on, which side is the alien on,
                       is the gun ready?) and plain code decides the move.
  --schema policy      the model picks the move itself (a conditional question:
                       it tends to fail zero-shot, kept here as a comparison).

Note on frameskip 3 (not 4): Space Invaders draws bullets on alternate frames,
so with an even frameskip you always land on the same kind of frame and the
enemy missiles become invisible.

Install:
    pip install gliner2 transformers peft accelerate
    pip install "gymnasium[atari]" ale-py ocatari imageio imageio-ffmpeg
Run:
    python space_invaders_gliner.py --policy rules  --render video
    python space_invaders_gliner.py --policy gliner --render video --verbose
"""
import argparse, time, statistics
from collections import Counter
from ocatari.core import OCAtari

# ---------- 1. Game state ----------
def cat(o):
    return getattr(o, "category", "").lower()

def read_state(objects, prev_bullets):
    objs = [o for o in objects if o is not None and cat(o) not in ("", "noobject")]
    player = next((o for o in objs if "player" in cat(o)), None)
    aliens = [o for o in objs if "alien" in cat(o)]
    # Coordinates, not objects: OCAtari may reuse object instances across frames.
    bullets = [(o.x, o.y) for o in objs if "bullet" in cat(o)]
    if player is None:
        return None, bullets
    px, py = player.x + player.w / 2, player.y

    def falling(b):  # a missile falls: its y grows compared with the previous step
        return any(abs(b[0] - p[0]) <= 4 and b[1] > p[1] for p in prev_bullets)
    missiles = [b for b in bullets if b[1] < py and falling(b)]
    my_shots = [b for b in bullets if b[1] < py and not falling(b)]

    missile = min(missiles, key=lambda b: (py - b[1]) + abs(b[0] - px), default=None)
    target = min(aliens, key=lambda a: abs(a.x + a.w / 2 - px), default=None)
    return {
        "n_bullets": len(bullets), "n_missiles": len(missiles),
        "missile_dx": None if missile is None else missile[0] - px,
        "missile_dy": None if missile is None else py - missile[1],
        "target_dx": None if target is None else (target.x + target.w / 2) - px,
        "gun_ready": len(my_shots) == 0,
    }, bullets

def missile_threat(s):
    return s["missile_dx"] is not None and s["missile_dy"] < 50 and abs(s["missile_dx"]) < 8

# ---------- 2. State -> text ----------
def describe(s):
    parts = []
    if missile_threat(s):
        if abs(s["missile_dx"]) < 3:
            parts.append("DANGER: an enemy missile is falling exactly ABOVE the ship.")
        else:
            side = "LEFT" if s["missile_dx"] < 0 else "RIGHT"
            parts.append(f"DANGER: an enemy missile is falling to the {side} of the ship.")
    else:
        parts.append("No missile is falling near the ship.")
    dx = s["target_dx"]
    if dx is None:
        parts.append("No aliens are visible.")
    elif abs(dx) < 4:
        parts.append("The nearest alien is exactly ABOVE the ship.")
    else:
        parts.append(f"The nearest alien is to the {'LEFT' if dx < 0 else 'RIGHT'} of the ship.")
    parts.append("The gun is ready." if s["gun_ready"] else "The gun is reloading.")
    return " ".join(parts)

# Schema A: the model reads, the code decides.
PERCEPTION = {
    "missile": {"labels": ["left", "right", "above", "none"],
             "prompt": "Where is the enemy missile relative to the ship?"},
    "alien": {"labels": ["left", "right", "above", "none"],
              "prompt": "Where is the nearest alien relative to the ship?"},
    "gun": {"labels": ["ready", "reloading"],
            "prompt": "Is the gun ready or reloading?"},
}

def combine(missile, alien, gun):
    """A handful of fixed rules turning the reading into (move, shoot)."""
    if missile == "left" or missile == "above":
        return "right", "no"
    if missile == "right":
        return "left", "no"
    if alien in ("left", "right"):
        return alien, "no"
    if alien == "above":
        return "stay", "yes" if gun == "ready" else "no"
    return "stay", "no"

# Schema B: the model picks the move itself (conditional question, the first attempt).
POLICY = {
    "move": {"labels": ["left", "right", "stay"],
             "prompt": "If there is danger, which way should the ship move to avoid the missile? "
                       "Otherwise, which way should it move to get under the nearest alien?"},
    "shoot": {"labels": ["yes", "no"],
              "prompt": "Is the nearest alien exactly above the ship and is the gun ready?"},
}

# ---------- 3. Policies ----------
class GlinerPolicy:
    def __init__(self, schema_name):
        import torch
        from gliner2 import AutoExtractor
        self.model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.schema_name = schema_name
        self.latencies = []

    def __call__(self, s):
        text = describe(s)
        schema = PERCEPTION if self.schema_name == "perception" else POLICY
        t0 = time.perf_counter()
        out = self.model.classify_text(text, schema)
        self.latencies.append((time.perf_counter() - t0) * 1000)
        if self.schema_name == "perception":
            move, shoot = combine(out["missile"], out["alien"], out["gun"])
            return move, shoot, f"{text}  -> read {out}"
        return out["move"], out["shoot"], text

def rule_policy(s):
    text = describe(s)
    if missile_threat(s):
        return ("right" if s["missile_dx"] <= 0 else "left"), "no", text
    dx = s["target_dx"]
    if dx is None:
        return "stay", "no", text
    if abs(dx) < 4:
        return "stay", ("yes" if s["gun_ready"] else "no"), text
    return ("left" if dx < 0 else "right"), "no", text

ACTION_NAME = {
    ("stay", "no"): "NOOP", ("stay", "yes"): "FIRE",
    ("left", "no"): "LEFT", ("left", "yes"): "LEFTFIRE",
    ("right", "no"): "RIGHT", ("right", "yes"): "RIGHTFIRE",
}

# ---------- 4. Game loop ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["gliner", "rules"], default="gliner")
    ap.add_argument("--schema", choices=["perception", "policy"], default="perception")
    ap.add_argument("--render", choices=["human", "video", "none"], default="none")
    ap.add_argument("--frameskip", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    render_mode = {"human": "human", "video": "rgb_array", "none": None}[args.render]
    env = OCAtari("ALE/SpaceInvaders-v5", mode="ram", hud=False, render_mode=render_mode,
                  frameskip=args.frameskip, repeat_action_probability=0.0)
    meanings = env.unwrapped.get_action_meanings()
    policy = GlinerPolicy(args.schema) if args.policy == "gliner" else rule_policy
    budget_ms = args.frameskip * 1000 / 60

    env.reset(seed=0)
    prev_bullets, score, frames = [], 0.0, []
    counts, mismatches = Counter(), Counter()
    agree = total = steps_bullets = steps_missiles = 0
    for step in range(args.max_steps):
        s, prev_bullets = read_state(env.objects, prev_bullets)
        if s:
            steps_bullets += s["n_bullets"] > 0
            steps_missiles += s["n_missiles"] > 0
            move, shoot, text = policy(s)
            if args.policy == "gliner":  # compare with the rules on the same state
                r_move, r_shoot, _ = rule_policy(s)
                total += 1
                if (move, shoot) == (r_move, r_shoot):
                    agree += 1
                else:
                    mismatches[f"model={ACTION_NAME[(move, shoot)]} rules={ACTION_NAME[(r_move, r_shoot)]}"] += 1
        else:
            move, shoot, text = "stay", "no", "ship not visible"
        name = ACTION_NAME[(move, shoot)]
        counts[name] += 1
        if args.verbose:
            print(f"{step:5d} | {name:9s} | {text}")
        _, reward, terminated, truncated, _ = env.step(meanings.index(name))
        score += reward
        if args.render == "video":
            frames.append(env.render())
        if terminated or truncated:
            break

    tag = args.policy + (f"_{args.schema}" if args.policy == "gliner" else "")
    print(f"\nPolicy={tag}  frameskip={args.frameskip}  steps={step + 1}  score={score:.0f}")
    print("Actions chosen:", dict(counts.most_common()))
    print(f"Steps with bullets detected: {steps_bullets}  |  with falling missiles: {steps_missiles}")
    if args.policy == "gliner":
        if total:
            print(f"Agreement with the rule-based policy: {100 * agree / total:.1f}%")
            print("Most frequent disagreements:", mismatches.most_common(5))
        lat = sorted(policy.latencies)
        if lat:
            print(f"Latency ms: mean={statistics.mean(lat):.1f}  p50={lat[len(lat)//2]:.1f}  "
                  f"p95={lat[int(len(lat)*0.95)]:.1f}  (budget: {budget_ms:.0f} ms)")
    if frames:
        import imageio
        fname = f"space_invaders_{tag}.mp4"
        imageio.mimsave(fname, frames, fps=60 // args.frameskip)
        print(f"Video saved: {fname}")
    env.close()

if __name__ == "__main__":
    main()

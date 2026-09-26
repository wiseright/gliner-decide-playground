"""
Space Invaders Copilot, v0.2 - talk to the ship.

A human copilot gives orders in plain English. GLiNER2.5-Decide translates each
order into a STRATEGY (mode / zone / focus) that stays active until the next
order arrives. Every frame, a rule-based player acts on the current strategy.

The model is called ONLY when an order arrives, like setting cruise control
once and letting the car hold the speed. And it is called in a worker thread, so
the game loop never blocks on inference: on a slow CPU the ship simply reacts a
few steps later instead of the whole game stuttering.

Command input (--input):
  keyboard  1/2/3 = preset phrases, 4 = free text, or just type the sentence. q = quit
  voice     push-to-talk with Whisper: press Enter, speak for a few seconds
            (1/2/3 still work as shortcuts)
  script    orders scheduled by game step, for Colab or for reproducible demos:
            --script "150:1; 600:2; 1100:Stay on the left and don't take risks"

Install:
  pip install gliner2 "gymnasium[atari]" ale-py ocatari imageio imageio-ffmpeg
  pip install faster-whisper sounddevice      # only for --input voice

Examples:
  python space_invaders_copilot.py --render human                      # local, keyboard
  python space_invaders_copilot.py --render human --input voice        # local, voice
  python space_invaders_copilot.py --render video --input script \\
      --script "100:1; 500:2; 900:3" --max-steps 1500                  # Colab
"""
import argparse, queue, sys, threading, time, statistics
from ocatari.core import OCAtari

# ============ 1. Commands and strategy ============
PRESETS = {
    "1": "Play it safe: stay behind the shields and dodge every missile.",
    "2": "Go aggressive! Chase the aliens and shoot as much as you can.",
    "3": "Hold the right side and take out the lowest aliens first.",
}

COMMAND_SCHEMA = {
    "mode": {"labels": ["aggressive", "defensive", "balanced"],
             "prompt": "What play style does the copilot ask for?"},
    "zone": {"labels": ["left", "center", "right", "anywhere"],
             "prompt": "In which part of the screen should the ship stay?"},
    "focus": {"labels": ["nearest_alien", "lowest_alien", "mystery_ship"],
              "prompt": "Which enemy should the ship target first?"},
}

DEFAULT_STRATEGY = {"mode": "balanced", "zone": "anywhere", "focus": "nearest_alien"}

class CommandInterpreter:
    """The only place GLiNER is used: free text -> strategy."""
    def __init__(self):
        import torch
        from gliner2 import AutoExtractor
        self.model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.latencies = []

    def __call__(self, text):
        t0 = time.perf_counter()
        strategy = self.model.classify_text(text, COMMAND_SCHEMA)
        self.latencies.append((time.perf_counter() - t0) * 1000)
        return strategy, self.latencies[-1]

# ============ 2. Interpretation worker (the model runs here, off the game loop) ============
def interpreter_worker(interpreter, raw_q, out_q):
    """Raw order text -> strategy, so the game loop never waits for the model."""
    while True:
        kind, text = raw_q.get()
        if kind == "quit":
            out_q.put(("quit", None))
            continue
        strategy, ms = interpreter(text)
        out_q.put(("strategy", (text, strategy, ms)))

# ============ 3. Command sources (they run in a thread, the game never stops) ============
def resolve(line):
    """'1'/'2'/'3' -> preset phrase, anything else is passed through as-is."""
    return PRESETS.get(line, line)

def print_menu(extra=""):
    print("\n=== COPILOT COMMANDS ===")
    for k, v in PRESETS.items():
        print(f"  {k} -> {v}")
    print("  4 -> free text (English)")
    if extra:
        print(f"  {extra}")
    print("  q -> quit\n")

def keyboard_source(q):
    print_menu()
    awaiting_free_text = False
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line.lower() == "q":
            q.put(("quit", None)); return
        if awaiting_free_text:
            q.put(("cmd", line)); awaiting_free_text = False
        elif line == "4":
            print("Type your command in English:"); awaiting_free_text = True
        elif line.startswith("4 "):
            q.put(("cmd", line[2:].strip()))
        else:
            q.put(("cmd", resolve(line)))

def voice_source(q, seconds=3.0, whisper_size="base.en"):
    """Push-to-talk: Enter -> record `seconds` seconds -> Whisper -> command."""
    import sounddevice as sd
    from faster_whisper import WhisperModel
    whisper = WhisperModel(whisper_size, device="auto", compute_type="default")
    print_menu(extra=f"Enter -> talk for {seconds:.0f} s")
    for raw in sys.stdin:
        line = raw.strip()
        if line.lower() == "q":
            q.put(("quit", None)); return
        if line:                             # text or 1/2/3: keyboard shortcuts
            q.put(("cmd", resolve(line))); continue
        print("Listening...")
        audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        segments, _ = whisper.transcribe(audio[:, 0], language="en")
        text = " ".join(s.text for s in segments).strip()
        print(f"Heard: '{text}'")
        if text:
            q.put(("cmd", text))

def parse_script(s):
    """'150:1; 600:Stay left' -> {150: preset phrase, 600: 'Stay left'}"""
    out = {}
    for part in filter(None, (p.strip() for p in (s or "").split(";"))):
        step, text = part.split(":", 1)
        out[int(step)] = resolve(text.strip())
    return out

# ============ 4. Game state ============
def cat(o):
    return getattr(o, "category", "").lower()

def center(o):
    return o.x + o.w / 2

def read_state(objects, prev_bullets):
    objs = [o for o in objects if o is not None and cat(o) not in ("", "noobject")]
    player = next((o for o in objs if "player" in cat(o)), None)
    bullets = [(o.x, o.y) for o in objs if "bullet" in cat(o)]  # coordinates, not objects
    if player is None:
        return None, bullets
    px, py = center(player), player.y

    def falling(b):
        return any(abs(b[0] - p[0]) <= 4 and b[1] > p[1] for p in prev_bullets)
    missiles = [b for b in bullets if b[1] < py and falling(b)]
    my_shots = [b for b in bullets if b[1] < py and not falling(b)]
    missile = min(missiles, key=lambda b: (py - b[1]) + abs(b[0] - px), default=None)
    mystery = next((o for o in objs if any(k in cat(o) for k in ("satellite", "ufo", "mother"))), None)

    return {
        "px": px,
        "aliens": [(center(o), o.y) for o in objs if "alien" in cat(o)],
        "shields": [center(o) for o in objs if "shield" in cat(o)],
        "mystery_x": None if mystery is None else center(mystery),
        "missile_dx": None if missile is None else missile[0] - px,
        "missile_dy": None if missile is None else py - missile[1],
        "gun_ready": len(my_shots) == 0,
    }, bullets

# ============ 5. Rule-based player, parameterised by the strategy ============
MODES = {  # dodge thresholds and firing-alignment tolerance
    "defensive":  {"dodge_dy": 70, "dodge_dx": 12, "align": 3},
    "balanced":   {"dodge_dy": 50, "dodge_dx": 8,  "align": 4},
    "aggressive": {"dodge_dy": 30, "dodge_dx": 5,  "align": 8},
}
ZONES = {"left": (0, 65), "center": (55, 105), "right": (95, 160), "anywhere": (0, 160)}  # Atari pixels (screen is 160 wide)

def pick_target(s, focus):
    if focus == "mystery_ship" and s["mystery_x"] is not None:
        return s["mystery_x"]
    if not s["aliens"]:
        return None
    if focus == "lowest_alien":
        return max(s["aliens"], key=lambda a: a[1])[0]
    return min(s["aliens"], key=lambda a: abs(a[0] - s["px"]))[0]

def play(s, strategy):
    p = MODES[strategy["mode"]]
    lo, hi = ZONES[strategy["zone"]]
    px = s["px"]

    # 1) dodge
    if s["missile_dx"] is not None and s["missile_dy"] < p["dodge_dy"] and abs(s["missile_dx"]) < p["dodge_dx"]:
        move = "right" if s["missile_dx"] <= 0 else "left"
        if (move == "right" and px > hi - 4) or (move == "left" and px < lo + 4):
            move = "left" if move == "right" else "right"   # stay inside the zone when possible
        return move, "no"

    # 2) choose where to go
    tx = pick_target(s, strategy["focus"])
    goal = (lo + hi) / 2 if tx is None else min(max(tx, lo), hi)
    if strategy["mode"] == "defensive" and not s["gun_ready"] and s["shields"]:
        goal = min(s["shields"], key=lambda x: abs(x - px))  # while reloading, slide back under a shield
        goal = min(max(goal, lo), hi)

    # 3) shoot
    shoot = "yes" if (s["gun_ready"] and tx is not None and abs(tx - px) < p["align"]) else "no"

    dx = goal - px
    move = "stay" if abs(dx) < 2 else ("left" if dx < 0 else "right")
    return move, shoot

ACTION_NAME = {
    ("stay", "no"): "NOOP", ("stay", "yes"): "FIRE",
    ("left", "no"): "LEFT", ("left", "yes"): "LEFTFIRE",
    ("right", "no"): "RIGHT", ("right", "yes"): "RIGHTFIRE",
}

# ============ 6. Game loop ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["keyboard", "voice", "script"], default="keyboard")
    ap.add_argument("--script", default="", help='e.g. "150:1; 600:Stay left and play safe"')
    ap.add_argument("--render", choices=["human", "video", "none"], default="human")
    ap.add_argument("--frameskip", type=int, default=3)   # 3, not 4: enemy missiles flicker on alternate frames
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--voice-seconds", type=float, default=3.0)
    args = ap.parse_args()

    interpreter = CommandInterpreter()
    interpreter("warm up")  # the first call is always the slowest

    render_mode = {"human": "human", "video": "rgb_array", "none": None}[args.render]
    env = OCAtari("ALE/SpaceInvaders-v5", mode="ram", hud=False, render_mode=render_mode,
                  frameskip=args.frameskip, repeat_action_probability=0.0)
    meanings = env.unwrapped.get_action_meanings()
    step_period = args.frameskip / 60        # wall-clock duration of one game step
    realtime = args.input != "script"        # with a human at the controls the game must run at real speed

    raw_q, q = queue.Queue(), queue.Queue()      # raw text in, interpreted strategy out
    if args.input == "keyboard":
        threading.Thread(target=keyboard_source, args=(raw_q,), daemon=True).start()
    elif args.input == "voice":
        threading.Thread(target=voice_source, args=(raw_q, args.voice_seconds), daemon=True).start()
    threading.Thread(target=interpreter_worker, args=(interpreter, raw_q, q), daemon=True).start()
    scripted = parse_script(args.script)

    strategy, log = dict(DEFAULT_STRATEGY), []
    env.reset(seed=0)
    prev_bullets, score, frames = [], 0.0, []
    print(f"Strategy: {strategy}")

    for step in range(args.max_steps):
        t_step = time.perf_counter()

        # --- scheduled orders: interpreted inline, on purpose, so that a
        #     --script demo is reproducible step by step (in script mode the
        #     game does not run at real speed anyway).
        if step in scripted:
            strategy, ms = interpreter(scripted[step])
            log.append((step, scripted[step], strategy, ms))
            print(f"[step {step}] '{scripted[step]}'\n    -> {strategy}   ({ms:.1f} ms)")

        # --- orders from keyboard or voice: already interpreted in the worker
        #     thread, so the game loop never blocks on the model.
        quit_game = False
        while not q.empty():
            kind, payload = q.get_nowait()
            if kind == "quit":
                quit_game = True
            else:
                text, strategy, ms = payload
                log.append((step, text, strategy, ms))
                print(f"[step {step}] '{text}'\n    -> {strategy}   ({ms:.1f} ms)")
        if quit_game:
            break

        # --- every frame: rules + current strategy ---
        s, prev_bullets = read_state(env.objects, prev_bullets)
        move, shoot = play(s, strategy) if s else ("stay", "no")
        _, reward, terminated, truncated, _ = env.step(meanings.index(ACTION_NAME[(move, shoot)]))
        score += reward
        if args.render == "video":
            frames.append(env.render())
        if terminated or truncated:
            break

        if realtime:  # keep the game running at its original speed
            time.sleep(max(0.0, step_period - (time.perf_counter() - t_step)))

    print(f"\nGame over  steps={step + 1}  score={score:.0f}")
    print(f"Commands: {len(log)}")
    lat = sorted(ms for *_, ms in log)
    if lat:
        print(f"Interpretation latency ms: mean={statistics.mean(lat):.1f}  max={lat[-1]:.1f}  "
              f"(one game step = {step_period * 1000:.0f} ms)")
    if frames:
        import imageio
        imageio.mimsave("space_invaders_copilot.mp4", frames, fps=60 // args.frameskip)
        print("Video saved: space_invaders_copilot.mp4")
    env.close()

if __name__ == "__main__":
    main()

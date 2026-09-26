# Contributing

Thanks for taking a look. This is a small companion repository for an article,
so the scope is deliberately narrow — but a few kinds of contribution are
genuinely valuable.

## Especially welcome

* **Benchmark reproductions on other hardware.** The §3 numbers come from a
  single free Colab T4 session. Results from other GPUs, other driver and
  library versions, or fp16 instead of fp32, are the most useful thing anyone
  can add. Open an issue with your table and your environment.
* **Corrections.** Wrong claim, wrong licence attribution, broken link,
  mistranslated Atari behaviour — all fair game, and please be blunt about it.
* **New decision-model examples** in the style of §2: a real use case, a dozen
  lines, no dependencies beyond `gliner2`.

## Less likely to be merged

* Turning the game scripts into a general-purpose RL framework. They are
  intentionally small and readable — the article walks through them line by line.
* Additions that require a paid API key to run.
* Large refactors that make the code diverge from the article's snippets.

## Ground rules

* Keep examples runnable on a **free Colab T4**. That constraint is the point.
* Comments and identifiers in **English**, matching the existing style.
* No Atari ROMs, no ROM links, no model weights committed to the repository.
* If you report a benchmark number, say what hardware and library versions
  produced it. An unlabelled number isn't a result.

## Reporting an issue

Include: what you ran, what you expected, what happened, your GPU (or CPU),
your Python version, and the versions of `gliner2`, `transformers` and `ale-py`.

## Licensing of contributions

By submitting a pull request you agree that your contribution is licensed under
the repository's [MIT License](LICENSE).

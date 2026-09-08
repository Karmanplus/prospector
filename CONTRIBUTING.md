# Contributing

Creating issues is welcome. If you are planning something bigger than a fix, open one first, since
a few of the constraints below are not obvious from reading the code.

## Setup

`pixi install`, then copy the example config library into place. The README covers both. Before
editing anything:

```bash
pixi run check
```

That runs ruff and the test suite, and both need to pass. The suite runs against `examples/configs`
rather than a real config library, so no test may use user-specific configurations.

## A few constraints

This is a screening tool, and the costly mistake is discarding a target that was actually
reachable. The reachability delta-v estimates therefore compute low on purpose, and none of them
may reject a target by itself. A guiding philosophy for this project is to not reject an answer
until an appropriate fidelity solver can more definitively show that something is unreachable.

Everything under `prospector/` is a plain importable library with no UI framework in it. `ui/` and
`worker.py` are wrappers over it. This separation should be maintained.

Engines, launch types, propellants, missions and sizing coefficients live in YAML and are read at
run time. Be wary of default values that may silently overwrite intended loads from these YAMLs and
prefer load failures over fallbacks.

Tests sit where their module sits, so `prospector/trades/pipeline/solve.py` is covered by
`tests/trades/pipeline/test_solve.py`.

Some of the physics here was tuned against measurements. If you change those numbers, pin them with
a test.

## Style

Match the file you are editing. Comments should explain the physics and the intent as the code
stands, written for somebody who has not seen it before. Derivations belong in `docs/physics.md`.

## Using AI tools

Writing code with an AI assistant is fine, with two conditions.

Anything it produces has to be checked by a test wherever a test is possible, rather than read over
and judged correct. The mistakes that matter here do not show up by inspection: a screening bound
that drops reachable targets, or a hardcoded default that stands in for real data and still returns
a plausible number.

Issues and pull requests have to be written by a person, in your own words, saying what you
actually ran. Generated summaries read fluently and tend to leave out the detail a reviewer needs,
and they make it hard to separate what you checked from what the tool assumed.

## Sending it

One branch per change, and keep it small enough to read in one sitting. Say in the description what
you changed and what you ran to check it.

## Licence

Everything here is under [Apache-2.0](LICENSE), and so is anything you contribute: section 5 of the
licence puts a submission under those terms unless you say otherwise, so there is nothing to sign.
You keep the copyright on what you wrote, and nothing is assigned to anyone else.

Apache-2.0 is permissive, so anyone can take this code into a closed or commercial product without
publishing anything back. Contributing here means accepting that, including the patent grant in
section 3, which is what stops a contributor later asserting a patent against people using the code
they contributed to.

Source files need no licence header. `LICENSE` and `NOTICE` at the root cover the tree, and both
have to travel with anything built from it.

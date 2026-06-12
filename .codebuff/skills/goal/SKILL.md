---
name: goal
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
description: Set a standing goal and iterate toward it across turns with a judge model and turn budget. Inspired by Hermes Agent's /goal (Ralph loop pattern). Use for tasks you want the agent to keep working on without re-prompting every turn.
---

# /goal — Standing Goal Loop

## Overview

`/goal <text>` sets a **standing objective** that persists across turns in `.codebuff/goal_state.json`. After each turn a lightweight judge prompt checks whether the goal is satisfied by the assistant's last response. If not, the skill emits a `↻ Continuing toward goal` continuation prompt that you can resend to keep the loop alive — until the goal is achieved, you pause or clear it, or the turn budget runs out.

This is a from-scratch port of the Ralph loop pattern adapted to Codebuff's single-turn CLI model. Where Hermes Agent runs an implicit judge + auto-continuation loop in one session, Codebuff requires the user to re-send (the skill emits the continuation prompt at the end of each turn so you can do this with one Enter).

## Trigger

| Command                       | What it does                                                            |
|-------------------------------|-------------------------------------------------------------------------|
| `/goal <text>`                | Set (or replace) the standing goal. Kicks off the first iteration.      |
| `/goal` or `/goal status`     | Show current goal, subgoals, status, and turns used.                   |
| `/goal pause`                 | Stop the loop without clearing the goal.                                |
| `/goal resume`                | Resume the loop (resets the turn counter back to zero).                 |
| `/goal clear`                 | Drop the goal + all subgoals entirely.                                  |
| `/goal budget <N>`            | Change the turn budget (default 20).                                    |
| `/subgoal <text>`             | Append a new acceptance criterion to the active goal.                   |
| `/subgoal`                    | Show the current numbered subgoal list.                                 |
| `/subgoal remove <N>`         | Remove the Nth subgoal (1-based).                                       |
| `/subgoal clear`              | Drop every subgoal but keep the original goal intact.                   |

## State File

All goal state persists to `.codebuff/goal_state.json` (created on first set). Schema:

```json
{
  "goal": "Free-form standing objective text",
  "subgoals": ["Optional criterion 1", "Optional criterion 2"],
  "status": "active|paused|done|cleared",
  "turns_used": 0,
  "max_turns": 20,
  "created_at": "2026-06-10T18:30:00Z",
  "updated_at": "2026-06-10T18:30:00Z",
  "history": [
    {"turn": 1, "verdict": "continue|done", "reason": "Judge rationale"}
  ]
}
```

`status: "done"` is set when the judge marks the goal complete. `paused` keeps the goal text but stops the loop. `cleared` removes the file.

## Workflow

### Step 0: Parse the subcommand

If `$ARGUMENTS` is empty → show `/goal status`. If it equals `pause|resume|clear|status` → run that subcommand. If it equals `budget <N>` → update `max_turns`. If it starts with a number → noop (status always shows budget). Otherwise → treat everything after `/goal ` as the new goal text.

### Step 1: Load + reconcile state

Read `.codebuff/goal_state.json` (if it exists). If `status` is `cleared` or the file is missing, treat as no active goal (the next `/goal <text>` will create one).

If state exists and `status` is `paused`, display the paused state and tell the user to `/goal resume`.

If state exists and `status` is `active` AND a new `/goal <text>` is being set, REPLACE the goal text and clear the subgoals (this is the same replace-and-clear semantics Hermes uses — see doc footnote on `/goal <new text>`). Confirm with the user before clobbering: `Existing goal will be replaced. Proceed? (yes/no)`.

If state exists and `status` is `done`, tell the user the previous goal is done and ask if they want to start a new one.

### Step 2: Set the goal (first invocation of `/goal <text>`)

1. Write a fresh state file with:
   - `goal` = the user's text
   - `subgoals` = []
   - `status` = `active`
   - `turns_used` = 0
   - `max_turns` = 20 (or whatever was previously set)
   - `created_at` / `updated_at` = now (use `Bash: date -u +%Y-%m-%dT%H:%M:%SZ`)
   - `history` = []
2. Print: `⊙ Goal set (20-turn budget): <goal text>`
3. Continue to Step 3 to do the first unit of work.

### Step 3: Execute one iteration

Treat the standing goal as your assignment. Do ONE unit of work per turn — make progress, then stop. Do NOT try to complete the entire goal in one shot (the loop exists so you can check in with the judge each turn).

Use the tools you have (Read/Write/Edit/Bash/Glob/Grep) to advance toward the goal. At the end of your response, ALWAYS output a structured footer — see Step 5.

### Step 4: Judge

After the unit of work, evaluate whether the goal is satisfied. The judge is **you, the same LLM**, performing a self-check. Be honest — do not mark done prematurely.

**Done** if:
- The deliverable is clearly produced (e.g. file written, test passes, command exits 0)
- The user's response explicitly confirms completion
- The goal is unachievable / blocked (treat as DONE with a block reason so we don't burn budget)

**Continue** if:
- The goal is partially advanced but more work remains
- You can identify the next concrete step

The judge prompt template (mental model — do not literally paste this; just think it):

```
GOAL: <goal text>
SUBGOALS (all must be met): <list>
LAST RESPONSE (last ~4 KB of your own turn):
<your output>
---
Verdict (strict JSON): {"done": <bool>, "reason": "<one-sentence rationale>"}
```

### Step 5: Output the loop footer

End every turn with EXACTLY this footer format. The user (or another tool) parses this to know whether to resend.

```
[goal] turn=<N>/<max> status=<active|paused|done> verdict=<continue|done> reason="<judge's one-sentence reason>"
```

Examples:

- `[goal] turn=3/20 status=active verdict=continue reason="Built first module, need to wire the second"`
- `[goal] turn=5/20 status=done verdict=done reason="All subgoals met, tests pass"`

If `verdict=done`, set `status=done` in the state file and stop. If `turns_used` reaches `max_turns`, set `status=paused` and print: `⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going, or /goal clear to stop.`

If `verdict=continue`, increment `turns_used` by 1 in the state file and print a continuation banner:

```
↻ Continuing toward goal (3/20): Built first module, need to wire the second.
→ Resend your last message (or just press ↑ + Enter) to run the next iteration.
```

The "resend" UX is Codebuff's equivalent of Hermes' auto-continuation loop. Each press of Enter = one turn of work + one judge check.

### Step 6: Handle subcommands

- `/goal status`: read state, print formatted status block (see Output Format below), then END (no work done this turn).
- `/goal pause`: set `status=paused`, print `⏸ Goal paused.`, END.
- `/goal resume`: set `status=active`, set `turns_used=0`, print `▶ Goal resumed (counter reset to 0).`, END.
- `/goal clear`: set `status=cleared` (or delete the file), print `✓ Goal cleared.`, END.
- `/goal budget <N>`: validate N is a positive int, set `max_turns=N`, print `✓ Turn budget set to N.`, END.
- `/subgoal <text>`: read state, append text to `subgoals`, print `✓ Subgoal added (N total).`, END.
- `/subgoal` (no args): print the numbered subgoal list, END.
- `/subgoal remove <N>`: pop the Nth (1-based) subgoal, END.
- `/subgoal clear`: empty the subgoals list (keep the original goal), END.

## Output Format (status)

```
=== GOAL STATUS ===
Goal:     <goal text>
Status:   <active|paused|done|cleared>
Turns:    <turns_used>/<max_turns>
Subgoals: <N> active
  1. <subgoal 1>
  2. <subgoal 2>
Created:  <created_at>
Updated:  <updated_at>
Last judge verdict: <continue|done> — <reason>
=====================
```

## Judge Failure Modes

Two failure modes to watch for (same as Hermes):

- **False negative** (judge says continue when actually done) — the user notices and `/goal clear`s, or sets a new goal.
- **False positive** (judge says done when work remains) — the user notices and resends, or refines the goal text with `/goal <more specific text>`.

If the judge gets it wrong, refine the goal text. Vague goals produce unreliable judge verdicts.

## When to Use /goal

Use it for tasks where the agent does multiple turns of work without re-prompting:
- "Fix every lint error in src/ and verify ruff check passes"
- "Port feature X from repo Y, including tests, and get CI green"
- "Build a small CLI to rename files by their EXIF dates, then test it against the photos/ folder"
- "Migrate the WC lineup populate job from cron to systemd timer"

Don't use it for single-turn tasks or quick questions — the overhead of the judge loop isn't worth it.

## Configuration

`max_turns` defaults to 20. Override globally by editing `.codebuff/goal_state.json` after the first `/goal` invocation, or per-goal with `/goal budget <N>`. The state file is the single source of truth — no separate config file is needed.

## Attribution

Inspired by Hermes Agent's `/goal` (https://hermes-agent.nousresearch.com/docs/user-guide/features/goals), which itself credits Eric Traut (OpenAI) and the Ralph loop pattern from Codex CLI 0.128.0. The core idea — keep a standing goal alive across turns with a judge + budget — is theirs. This Codebuff port is independent: state lives in `.codebuff/goal_state.json` instead of SessionDB, the judge is the same LLM doing a self-check (no auxiliary client), and the auto-continuation is user-driven (resend) instead of adapter-FIFO queued.

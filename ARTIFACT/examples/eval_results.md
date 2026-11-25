# Synthetic evaluation (400 inputs x 5 replays = 2000 runs, seed 7)

All numbers are synthetic and come from the bundled simulator.

## Where runs split and why

Forks scored against known causes: 283

| Method | Correct cause |
|---|---|
| Earliest evidence change (this tool) | 270/283 (95.4%) |
| Diff at the fork step only | 245/283 (86.6%) |

| True cause | Forks | Earliest change | Fork step only |
|---|---|---|---|
| deadline_hit | 10 | 10 | 10 |
| input | 17 | 17 | 0 |
| model | 19 | 19 | 19 |
| model,prompt_version | 9 | 9 | 9 |
| prompt_version | 22 | 22 | 22 |
| sampling | 135 | 135 | 135 |
| tool:orders_api | 29 | 29 | 21 |
| tool:search_kb | 12 | 12 | 12 |
| unrecorded | 30 | 17 | 17 |

In-process stamp on the root span names the right cause for 274/287 divergent runs.

## Sampling at the same budget (290 of 2000 traces kept)

| | Head sampling | Tail sampling with hints |
|---|---|---|
| Traces kept | 290 | 290 |
| Spans kept | 1671 | 1803 |
| JSON bytes kept | 934809 | 1040530 |
| Failed or stopped runs kept | 2/14 | 14/14 |
| Runs on rare paths kept | 9/44 | 29/44 |
| Rare paths with at least one example | 5/10 | 10/10 |
| Splits caused by a change you can explain | 7/149 | 137/149 |
| Splits caused by model sampling you can explain | 4/138 | 16/138 |

## Completeness checks

Injected incomplete traces 37, flagged 37, caught 37, false alarms 0.

## Size of the extra attributes

agent_state attributes add 66.6% to exported JSON bytes (3928522 to 6543738). Tail sampling keeps 26.5% of the bytes you would store by keeping every trace without them.

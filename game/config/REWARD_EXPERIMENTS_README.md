# Agent × Agent reward-comparison experiments

These configs reproduce the 5 reward functions from the QMIX
(`Xindictus/coop-marl-maze`) experiments inside this SAC codebase, so the two
algorithms can be compared in the same environment. Each config runs the
**agent × agent** setup (`mode: no_tl_two_agents`).

## Config files

| # | Reward                       | Config file                                          |
|---|------------------------------|------------------------------------------------------|
| 1 | Simple                       | `config_sac_two_agents_1_simple.yaml`                |
| 2 | Goal Distance                | `config_sac_two_agents_2_goal_distance.yaml`         |
| 3 | Progress Distance            | `config_sac_two_agents_3_progress_distance.yaml`     |
| 4 | Progress Distance + stalling | `config_sac_two_agents_4_progress_with_stalling.yaml`|
| 5 | Speed Stalling               | `config_sac_two_agents_5_speed_stalling.yaml`        |

## Commands

Run from the repo root (the trainer entry point is `sac_maze3d_train.py`):

```bash
python sac_maze3d_train.py --config game/config/config_sac_two_agents_1_simple.yaml                 --participant two_agents_simple
python sac_maze3d_train.py --config game/config/config_sac_two_agents_2_goal_distance.yaml          --participant two_agents_goal_distance
python sac_maze3d_train.py --config game/config/config_sac_two_agents_3_progress_distance.yaml      --participant two_agents_progress_distance
python sac_maze3d_train.py --config game/config/config_sac_two_agents_4_progress_with_stalling.yaml --participant two_agents_progress_stalling
python sac_maze3d_train.py --config game/config/config_sac_two_agents_5_speed_stalling.yaml         --participant two_agents_speed_stalling
```

Everything that matters is set in the config files, so no extra CLI flags are
needed. `--seed <int>` can be added for reproducibility (default 4213).


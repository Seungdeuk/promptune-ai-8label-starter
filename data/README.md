# Dataset contract

Use one row per prompt.

- `1` = 보완 필요(누락)
- `0` = 충분함 또는 해당 요청에서 추가 필요 없음

Required columns:

`id,text,missing_task,missing_audience,missing_context,missing_format,missing_tone,missing_length,missing_constraint,missing_example`

Do not add the old 5-label test data here. After the team finishes the 30-sentence calibration, create agreed Ground Truth first, then build train/validation/test files.

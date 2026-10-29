# 自动进度快照

更新时间：2026-10-30T06:32:41.930375+08:00
当前阶段：confirm；状态：running

|阶段|完整病例|病例分母|预测记录|
|---|---:|---:|---:|
|confirm|328|5126|2624|
|official_test|0|7458|0|
|med|0|852|0|

长上下文补跑：{"time": 1793313149.5891976, "status": "running", "phase": "confirm", "phases": {"confirm": {"target_cases": 107, "complete_cases": 2, "pending_cases": 105, "original_records_preserved": 856}, "official_test": {"target_cases": 0, "complete_cases": 0, "pending_cases": 0, "original_records_preserved": 0}, "med": {"target_cases": 0, "complete_cases": 0, "pending_cases": 0, "original_records_preserved": 0}}, "active_jobs": [{"gpu": 3, "pid": 797419, "cases": 4, "capacity": 65536, "elapsed_seconds": 421.5809783935547}], "blocked_jobs": [], "unsupported_cases": 0, "probe_complete": true, "controller_alive": false, "status_age_seconds": 12.1}

预计剩余时间：{"context_replay_not_yet_included": true, "basis": "sum of each live worker completed-case increments divided by current actual elapsed time including in-flight batches; future guideline mix may differ", "mega_cases_per_hour": 55.92, "remaining_mega_cases": 12256, "remaining_mega_days_at_current_rate": 9.13, "approximate_overall_days_range": [6.4, 13.8]}

本文件只发布汇总进度，不包含病例、提示或模型原文。实验由服务器原调度器运行，本发布程序不启动或重启实验。

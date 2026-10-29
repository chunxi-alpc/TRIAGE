# 自动进度快照

更新时间：2026-10-30T07:14:32.681692+08:00
当前阶段：confirm；状态：priority_context_replay

|阶段|完整病例|病例分母|预测记录|
|---|---:|---:|---:|
|confirm|328|5126|2624|
|official_test|0|7458|0|
|med|0|852|0|

长上下文补跑：{"time": 1793315663.4004695, "status": "running", "phase": "confirm", "phases": {"confirm": {"target_cases": 107, "complete_cases": 33, "pending_cases": 74, "original_records_preserved": 856}, "official_test": {"target_cases": 0, "complete_cases": 0, "pending_cases": 0, "original_records_preserved": 0}, "med": {"target_cases": 0, "complete_cases": 0, "pending_cases": 0, "original_records_preserved": 0}}, "active_jobs": [{"gpu": 6, "pid": 802191, "cases": 4, "capacity": 65536, "elapsed_seconds": 365.03775572776794}, {"gpu": 1, "pid": 801150, "cases": 4, "capacity": 65536, "elapsed_seconds": 881.7719922065735}, {"gpu": 5, "pid": 801837, "cases": 4, "capacity": 65536, "elapsed_seconds": 577.8984818458557}, {"gpu": 0, "pid": 801492, "cases": 4, "capacity": 65536, "elapsed_seconds": 760.1244201660156}], "blocked_jobs": [], "unsupported_cases": 0, "probe_complete": true, "known_queue_eta": {"completed_cases": 33, "cases_per_gpu_hour": 13.32, "remaining_cases": 74, "remaining_hours_at_current_cards": 1.39, "basis": "actual elapsed GPU-job time including model loading and in-flight cases; current discovered queue only; future overflow cases not included"}, "priority_cohort": {"target_cases": 107, "complete_cases": 33, "remaining_cases": 74, "allowed_gpus": [0, 1, 5, 6]}, "controller_alive": true, "status_age_seconds": 9.0}

预计剩余时间：{}

本文件只发布汇总进度，不包含病例、提示或模型原文。实验由服务器原调度器运行，本发布程序不启动或重启实验。

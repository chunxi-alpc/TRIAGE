# Codex02 E5 实际完成结果

本报告只收录已有完成记录与真实预测支持的单元。正式训练固定E1/Dataset A 2540对，Qwen3-8B；源训练端点4637。未重训ASTER，未重评NLI500或MedPIC188。

[新增E1结果统一表](section_4_5_new_e1.csv)保存逐方法结果和95%区间来源；[运行状态与完成文件](new_results_index.json)单列。原有历史13行结果继续保存在section_4_5_data.csv及对应原始目录；不把历史旧训练行冒充E1。

LoReFT沿用二元监督交叉熵的单层任务适配；LoRA采用单层rank1完整词表SFT；LoFiT采用Qwen架构适配的两阶段SFT，部署和选择阶段参数分开报告。LEACE-pair共享E1选头监督、使用配对概念编码，明确作为任务适配控制；ITI和SEA也明确沿用各自任务适配。它们不冒称未改动的官方原配置。

所有40来源、806自然路径结果均有历史使用记录；部分CPU控制另覆盖2418干扰路径，分母分别保存。没有将806替代官方7458。外部新候选由Codex03单列，此处未运行方法没有数字。

|方法|A/P RMS|C/F修正RMS|自然路径全对|
|---|---:|---:|---:|
|temperature_e1_endpoint_nll|0.382741|31.011310|768/806|
|intercept_e1_endpoint_nll|2.628152|0.000000|763/806|
|affine_e1_endpoint_nll|0.389380|30.919624|769/806|
|linear_readout_e1_last_attention_ce|0.778096|29.789956|766/806|
|lora_SFT_E1_seed11|2.434310|18.734697|756/806|
|lora_SFT_E1_seed22|2.310940|14.842368|774/806|
|lora_SFT_E1_seed33|2.517128|12.612368|737/806|
|lofit_SFT_E1_seed11|2.627258|0.042992|768/806|
|lofit_SFT_E1_seed22|2.626904|0.046250|769/806|
|lofit_SFT_E1_seed33|2.627998|0.057522|769/806|
|LoReFT_binary_task_adaptation_E1_seed11|2.131666|13.432439|764/806|
|LoReFT_binary_task_adaptation_E1_seed22|1.851528|14.745472|772/806|
|LoReFT_binary_task_adaptation_E1_seed33|1.867888|18.838354|650/806|
|leace_pair_E1|2.634580|0.556300|767/806|
|iti_style_E1|1.103956|31.905203|321/806|
|sea_adaptation_E1|2.151922|9.697613|768/806|
|CC|2.628152|0.000000|769/806|
|strong_evidence_prompt|2.747538|2.102427|772/806|

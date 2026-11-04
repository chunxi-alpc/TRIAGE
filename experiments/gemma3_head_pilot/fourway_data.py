"""Constructed single-fact development pairs; no original patient text is copied."""
import copy
import json
from collections import Counter
from pathlib import Path
from .core import rows, write_json, swap_actions

# Conditions and source trees come from the registered development split.
# These literal records are fixed before any model predictions are inspected.
# Each tuple is (source number, target node, fact type, shared record, false fact, true fact).
PAIR_SPECS = [
 ("1170", "G0", "diagnosis",
  "An adult was assessed after a fall. Adequate-standard hip X-rays are negative.",
  "After clinical examination, hip fracture is not suspected.",
  "After clinical examination, hip fracture remains suspected."),
 ("1170", "G1", "time",
  "An adult has a clinically suspected hip fracture despite adequate negative hip X-rays. There is no contraindication to MRI.",
  "The earliest available MRI appointment is in 36 hours.",
  "The earliest available MRI appointment is in 12 hours."),
 ("1372", "G0", "test_result",
  "患者目前已完成新冠核酸检测，抗原检测结果为阴性。",
  "本次核酸检测结果为阴性。", "本次核酸检测结果为阳性。"),
 ("1372", "G1", "time",
  "患者本次新冠核酸检测阳性。", "本次症状首次出现距今8天。", "本次症状首次出现距今6天。"),
 ("1389", "G4", "count",
  "患者有高尿酸血症，3年前曾有明确痛风发作。", "当前皮下痛风石数量为0个。", "当前皮下痛风石数量为1个。"),
 ("1389", "G8", "count",
  "患者3年前已有明确痛风发作，目前无皮下痛风石，影像学无痛风所致损伤。",
  "过去12个月内痛风发作1次。", "过去12个月内痛风发作2次。"),
 ("184", "G0", "age",
  "患者为女性，现进行宫颈癌筛查评估。", "当前年龄为24周岁。", "当前年龄为25周岁。"),
 ("184", "G1", "age",
  "患者为女性，现进行宫颈癌筛查评估。", "当前年龄为34周岁。", "当前年龄为35周岁。"),
 ("2324", "G1", "diagnosis",
  "患者拟接受择期非心脏手术，既往无心肌梗死及冠脉支架植入史。",
  "本次术前评估已明确排除冠心病。", "本次术前评估已明确诊断冠心病。"),
 ("2324", "G4", "time",
  "患者有冠心病，接受PCI并植入药物洗脱支架，现拟接受择期中高危非心脏手术，术前计划中断双联抗血小板治疗。",
  "距最近一次药物洗脱支架植入为11个月。", "距最近一次药物洗脱支架植入为13个月。"),
 ("2770", "G1", "risk_category",
  "患者经病理确诊甲状腺乳头状癌，已完成初次手术治疗。",
  "依据术后病理及临床资料，DTC复发风险评估为低危。", "依据术后病理及临床资料，DTC复发风险评估为高危。"),
 ("2770", "G2", "diagnosis",
  "患者确诊分化型甲状腺癌，已完成初次手术，术后DTC复发风险评估为高危。",
  "全面影像及病理评估未发现远处转移。", "病理已证实肺部远处转移。"),
 ("348", "G1", "finding",
  "Specialist prenatal ultrasound has confirmed congenital solitary kidney in the fetus.",
  "Specialist fetal ultrasound shows no extrarenal malformations.",
  "Specialist fetal ultrasound shows an extrarenal malformation."),
 ("348", "G2", "test_performed",
  "Prenatal specialist ultrasound diagnosed congenital solitary kidney and an extrarenal malformation. The baby has now been born.",
  "Neonatal/postnatal ultrasound to confirm the kidney diagnosis has not yet been performed.",
  "Neonatal/postnatal ultrasound to confirm the kidney diagnosis has been performed."),
 ("365", "G1", "diagnosis",
  "患者有明确冠心病，属于心血管疾病患者，无难治性高血压，无肺动脉高压。",
  "心律评估明确排除心房颤动。", "心律评估明确诊断心房颤动。"),
 ("365", "G2", "measurement",
  "患者有冠心病及已确诊心房颤动，存在日间嗜睡，已完成连续6小时PSG，阻塞性事件占60%。",
  "睡眠监测AHI为4次/h。", "睡眠监测AHI为6次/h。"),
 ("604", "G0", "age",
  "患者正在接受脂肪肝相关评估。", "当前年龄为15周岁。", "当前年龄为17周岁。"),
 ("604", "G1", "measurement",
  "患者45岁。肝活检无炎症及纤维化，已排除大量饮酒、长期促脂肪形成药物及单基因遗传紊乱等其他肝脂肪变原因。",
  "活检显示肝脂肪变累及4%的肝细胞。", "活检显示肝脂肪变累及6%的肝细胞。"),
 ("948", "G1", "measurement",
  "患者在新冠病毒感染后仍有头痛。本次体温以外没有其他提示仍处发热阶段的记录。",
  "当前测得体温为37.1℃。", "当前测得体温为37.3℃。"),
 ("948", "G6", "symptom",
  "患者为36岁男性，新冠病毒感染后仍有头痛，体温已恢复至37.0℃，生命体征稳定，无重大基础疾病，无呼吸困难或缺氧，无黄脓痰，也无其他提示合并疾病、细菌感染或沉默性缺氧的表现。",
  "当前无长期反复咳嗽咳痰。", "当前有长期反复咳嗽咳痰。"),
]

def common_history(guideline, gid):
    groups = {g["group_id"]: g for g in guideline["groups"]}
    result, current, seen = [], gid, set()
    while groups[current].get("parent_group_id") is not None:
        if current in seen:
            raise ValueError("Cyclic guideline")
        seen.add(current)
        group = groups[current]
        result.append(group["parent_rule_id"])
        current = group["parent_group_id"]
    if current != "G0":
        raise ValueError("Target does not descend from G0")
    return list(reversed(result))

def validate_groups(groups):
    if len(groups) != 20 or len({g["group_id"] for g in groups}) != 20:
        raise ValueError("Expected exactly 20 unique development groups")
    counts = Counter(g["source_id"] for g in groups)
    if len(counts) != 10 or set(counts.values()) != {2}:
        raise ValueError("Expected two groups for each of ten development sources")
    for group in groups:
        if group["history"] != common_history(group["guideline"], group["gid"]):
            raise ValueError("Incorrect common prefix")
        target = next(g for g in group["guideline"]["groups"] if g["group_id"] == group["gid"])
        if {r["rule_id"] for r in target["rules"]} != {"R"+group["gid"][1:]+"Y", "R"+group["gid"][1:]+"N"}:
            raise ValueError("Target is not binary")
        if len(group["facts"]) != 2 or group["facts"][0] == group["facts"][1]:
            raise ValueError("Two distinct facts are required")
        swapped = swap_actions({"guideline": group["guideline"]})["guideline"]
        for before, after in zip(group["guideline"]["groups"], swapped["groups"]):
            if [r["condition"] for r in before["rules"]] != [r["condition"] for r in after["rules"]]:
                raise ValueError("Action view changed a condition")
    return counts

def prepare(pilot_root, output_root):
    pilot_root, output_root = Path(pilot_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "groups.jsonl"
    if path.exists():
        groups = rows(path)
        validate_groups(groups)
        return groups
    guides = {}
    for case in rows(pilot_root/"data/development.jsonl"):
        guides.setdefault(case["source_id"], {"guideline": case["guideline"], "question": case["question"]})
    groups = []
    for source, gid, fact_type, shared, negative, positive in PAIR_SPECS:
        source_id = "train_val:"+source
        item = guides[source_id]
        group_id = "fourway-dev-"+source+"-"+gid
        groups.append(dict(group_id=group_id, source_id=source_id, gid=gid,
                           history=common_history(item["guideline"], gid),
                           fact_type=fact_type, shared_record=shared, facts=[negative, positive],
                           guideline=item["guideline"], question=item["question"]))
    counts = validate_groups(groups)
    path.write_text("".join(json.dumps(g, ensure_ascii=False)+"\n" for g in groups))
    write_json(output_root/"data_manifest.json", dict(
        split="development", groups=len(groups), sources=len(counts), groups_per_source=dict(counts),
        input_cells=len(groups)*4, fact_types=dict(Counter(g["fact_type"] for g in groups)),
        construction="Literal single-fact constructed records using the ten registered development guidelines; original patient narratives are not copied.",
        labels="fact state 0 -> N; fact state 1 -> Y; unchanged across valid action views",
        evaluation="Fixed target node and common valid ancestor prefix; not autonomous full-path execution.",
        selection="All source/node/fact specifications fixed before viewing model predictions.",
        independent_confirmation_run=False))
    return groups

def make_case(group, state):
    return dict(case_id=group["group_id"]+":state"+str(state), source_id=group["source_id"],
                patient=group["shared_record"]+"\n"+group["facts"][state],
                guideline=group["guideline"], question=group["question"])

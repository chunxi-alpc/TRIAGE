"""Executable complete conditions and fixed factual construction specifications."""
import copy
import json
from common import CONFIG, DATA, read_rows, write_rows

def compare(field, operator, value):
    return {"op": operator, "field": field, "value": value}

def between(field, low, high):
    return {"op": "and", "args": [compare(field, ">=", low), compare(field, "<=", high)]}

def all_of(*args):
    return {"op": "and", "args": list(args)}

def any_of(*args):
    return {"op": "or", "args": list(args)}

def negate(arg):
    return {"op": "not", "arg": arg}

def flag(field):
    return compare(field, "==", True)

def evaluate(expression, facts):
    op = expression["op"]
    if op == "and":
        return all(evaluate(arg, facts) for arg in expression["args"])
    if op == "or":
        return any(evaluate(arg, facts) for arg in expression["args"])
    if op == "not":
        return not evaluate(expression["arg"], facts)
    a, b = facts[expression["field"]], expression["value"]
    return {"<": lambda: a < b, "<=": lambda: a <= b,
            ">": lambda: a > b, ">=": lambda: a >= b,
            "==": lambda: a == b, "!=": lambda: a != b}[op]()

def fields(expression):
    if expression["op"] in ("and", "or"):
        return set().union(*(fields(arg) for arg in expression["args"]))
    if expression["op"] == "not":
        return fields(expression["arg"])
    return {expression["field"]}

def make_specs(nodes, old_inputs):
    """The existing 50 IDs are the entire pool; never add replacement nodes."""
    ids = {r["node_id"] for r in old_inputs if r["population"] == "constructed"}
    selected = [n for n in nodes if n["node_id"] in ids]
    specs = {}
    def define(nid, metric, unit, values, expression, fixed=(), constraints=(), note=""):
        specs[nid] = dict(
            node_id=nid, eligible=True, variable="measurement", metric=metric, unit=unit,
            values=dict(zip(("below", "above", "same_side"), values)),
            predicate=expression,
            fixed_facts={key: value for key, value, sentence in fixed},
            fixed_fact_sentences={key: sentence for key, value, sentence in fixed},
            ancestor_constraints=list(constraints), note=note,
            neutral_background="The consultation record was entered in the electronic form.",
            cue_background="The patient or caregiver expresses a preference for active treatment.",
            neutral_condition=next(n["condition"] for n in selected if n["node_id"] == nid),
            wording_changed=False,
            background_policy="Only nonclinical detail or treatment preference varies.",
            ancestor_policy="The exact original ancestor conditions and outcomes are supplied unchanged.")
    def reject(nid, reason):
        specs[nid] = dict(node_id=nid, eligible=False, reason=reason)
    c = lambda op, val: compare("measurement", op, val)
    r = lambda low, high: between("measurement", low, high)
    nids = {i:n["node_id"] for i,n in enumerate(selected)}
    # IDs, not row order, determine rules.
    d = lambda prefix, index: prefix + ":" + str(index)
    p = "d8d6f57b62b5893ca0efe91c59e3470c"
    reject(d(p,26), "Ancestor age >=3 is false; crossing the current age <3 condition would change that ancestor.")
    define(d(p,6), "Age", "years", [2,4,5], c(">=",3))
    p = "2c44dbfd26d455b456beca2920543d6d"
    define(d(p,74), "Tumor mutational burden (TMB)", "mut/Mb", [9,11,12],
           any_of(flag("msi_high"), flag("dmmr"), c(">=",10)),
           [("msi_high",False,"MSI testing is not MSI-H (microsatellite stable)."),
            ("dmmr",False,"Mismatch repair is proficient (pMMR); dMMR is absent.")])
    define(d(p,15), "Age at diagnosis", "years", [39,41,42], c("<",40))
    define(d(p,17), "Age at diagnosis", "years", [49,51,52], r(40,50),
           constraints=[c(">=",40)], note="Inclusive 40-50 range; crossing its upper bound preserves ancestor age >=40.")
    p = "83cf1b073d03d1221b88fcd00c45f468"
    define(d(p,2), "ECOG performance status", "points", [0,2,3],
           all_of(r(0,1), flag("drainage"), flag("nutrition")),
           [("drainage",True,"Biliary drainage is good."),
            ("nutrition",True,"Nutritional intake is adequate.")])
    define(d(p,10), "Tumor mutational burden", "mut/Mb", [9,11,12], c(">=",10))
    p = "f8b2e3dcad5c734d8a6a9641fe410be7"
    define(d(p,1), "Estimated life expectancy", "years", [9,11,12], c(">=",10))
    for index in (8,30):
        define(d(p,index), "Age", "years", [34,36,37], c(">=",35))
    p = "ac29aff693f40424d1b43cf8b92bd776"
    for index in (4,22):
        define(d(p,index), "Bone marrow blast percentage", "%", [19,21,22],
               any_of(c(">=",20),compare("peripheral_blasts",">=",20)),
               [("peripheral_blasts",1,"Peripheral blood blasts are 1%.")],
               note="Blast phase is defined by the supplied >=20% criterion; no fixed blast-phase diagnosis is inserted.")
    define(d(p,18), "Bone marrow blast percentage", "%", [9,10,11],
           any_of(r(10,19),between("peripheral_blasts",10,19)),
           [("peripheral_blasts",1,"Peripheral blood blasts are 1%.")],
           constraints=[c("<",20)],
           note="The complete 10-19% range is retained and the ancestor excluding >=20% remains true.")
    p = "3a5e6d25d0dc9e2820526eb29571d60a"
    define(d(p,20), "Feasible interval from ART initiation to planned cancer therapy", "days", [6,8,9],
           any_of(flag("immediate_art"),c(">=",7),flag("urgent_exception")),
           [("immediate_art",False,"ART cannot be started immediately; its earliest feasible start is scheduled."),
            ("urgent_exception",False,"There is no urgent cancer-treatment timing exception.")],
           note="Immediate initiation and the urgency exception are both fixed false.")
    reject("bcd0d261305062272804fc06f5ebc465:37",
           "CD2/CD25/CD30 expression is categorical; the old 20% threshold is not supplied by this condition or its quoted source.")
    define("616e0a329049dfda55e76d9c7798ad91:13", "血清 IgM", "mg/dL", [5999,6001,6002],
           all_of(negate(flag("symptoms")),c(">",6000)),
           [("symptoms",False,"患者无症状。")])
    p = "4d89a2fdefa842e7abefa3bc220ba6cf"
    define(d(p,4), "病情评分", "分", [0,2,3],
           all_of(r(0,1),negate(flag("organ_disease"))),
           [("organ_disease",False,"无明显心、肺、肾、脑等重要器官器质性病变。")])
    reject(d(p,8), "The condition requires resection-specific ICG limits and an integrated reserve assessment; the old universal 0.1 cutoff does not encode it.")
    reject(d(p,12), "The rewritten condition says residual >=60% but equates it with resection <40%, disagreeing at equality; varying resection also changes an ancestor's resection-dependent reserve assessment. No unambiguous single-indicator pair is retained.")
    define("a9d93ad3125e6ef7640f378cbd3b17c7:9", "ECPR开始前CPR持续时间", "min", [59,61,62],
           c("<=",60), constraints=[c(">",40)])
    reject("e9f783677c8f87a9a25f341abd61c819:5",
           "Excluding all listed alternative causes is a conjunction, not a numerical count >=8; no original numerical threshold exists.")
    p = "316fa80cee731f90bdf32d1dfabe9754"
    define(d(p,13), "阿达木单抗皮下注射的给药间隔", "周", [1,2,2.5],
           all_of(flag("adalimumab_first"),flag("subcutaneous"),c("==",2)),
           [("adalimumab_first",True,"已选择阿达木单抗作为首选TNF抑制剂。"),
            ("subcutaneous",True,"每次给药采用皮下注射。")],
           note="An exact every-two-week schedule is categorical equality, not a directional threshold; excluded from primary crossing analysis.")
    specs[d(p,13)]["eligible"] = False
    specs[d(p,13)]["reason"] = "The original condition requires an exact every-two-week schedule, not a numerical threshold with a same-side control."
    define(d(p,34), "JIA起病年龄", "岁", [6,8,9],
           all_of(flag("ana_positive"),any_of(c("<",7),compare("duration","<=",4)),
                  any_of(flag("oligo"),flag("rf_negative_poly"),flag("psoriatic"),flag("undifferentiated"))),
           [("ana_positive",True,"ANA检测阳性。"),
            ("duration",5,"JIA病程为5年。"),
            ("oligo",True,"JIA分型为少关节炎型。"),
            ("rf_negative_poly",False,"不是RF阴性多关节炎型。"),
            ("psoriatic",False,"不是银屑病性关节炎。"),
            ("undifferentiated",False,"不是未分化关节炎。")])
    define(d(p,2), "前房细胞分级", "+", [0,0.5,1],
           any_of(c(">=",0.5),flag("other_inflammatory_signs")),
           [("other_inflammatory_signs",False,"未见其他眼部炎症体征。")])
    p = "176178ab78b4a90667d3313fc60b9e2b"
    reject(d(p,18), "Ancestors require age outside 16-60 and age not >60, which entails age <16; a crossing pair cannot preserve both.")
    define(d(p,12), "患者年龄", "岁", [15,61,62], c(">",60),
           constraints=[negate(r(16,60))],
           note="A wide crossing (15 versus 61) is necessary to keep the ancestor outside 16-60 true; reported separately.")
    p = "62fd1f5e51dded5bbd254e0aa71cf2fe"
    define(d(p,10), "尿液pH", "", [6.4,6.6,6.7], c(">",6.5))
    define(d(p,17), "结石直径", "mm", [19,21,22],
           all_of(flag("renal_stone"),r(10,20)),
           [("renal_stone",True,"结石位于肾内，不是输尿管结石。")],
           constraints=[c(">=",10)],
           note="Crosses the upper end, preserving the earlier no renal stone <10 mm condition.")
    p = "361d70abe387a692496e1396ec2b9bb0"
    define(d(p,1), "腹痛开始至血便出现的时间", "小时", [23,25,26],
           all_of(flag("elderly"),flag("acute_cramping_pain"),
                  any_of(flag("red_stool"),flag("dark_stool"),flag("bloody_diarrhea")),c("<=",24)),
           [("elderly",True,"患者为72岁。"),
            ("acute_cramping_pain",True,"患者出现急性痉挛样腹痛。"),
            ("red_stool",True,"出现鲜红色血便。"),
            ("dark_stool",False,"无暗红色血便。"),
            ("bloody_diarrhea",False,"无血性腹泻。")])
    define(d(p,22), "已开始的预防性抗生素计划疗程", "天", [1,2,3],
           all_of(flag("started"),r(2,3),flag("vasodilator_considered")),
           [("started",True,"已开始预防性抗生素治疗。"),
            ("vasodilator_considered",True,"已考虑应用扩血管药物。")],
           note="The duration is the planned short course, not elapsed treatment time.")
    define("a5601ad58cbdfd25f69c459332318a3f:3", "距最近一次服用NOACs的时间", "小时", [47,49,50],
           all_of(flag("history_known"),negate(c("<=",48))),
           [("history_known",True,"用药记录完整，可明确最近一次服药时间；此后没有再次服用NOACs。")])
    p = "d9906cd6be177a867400393324954892"
    for index,upper in ((13,30),(6,20)):
        define(d(p,index), "血小板计数", "×10^9/L", [9,11,12], c("<",10),
               constraints=[c("<",100),c("<",50),c("<",upper)])
    reject(d(p,21), "Ancestor already requires platelets <50; crossing the same <50 condition changes the ancestor.")
    define("afd5d03b38791ef112c4c90e963feed0:14", "血清 C 反应蛋白（CRP）", "mg/L", [19,21,22],
           all_of(negate(flag("other_etiology")),c(">",20)),
           [("other_etiology",False,"已排除其他可导致CRP升高的病因。")])
    p = "5aed0f29fa4b2754f2a5558f62c531ef"
    define(d(p,1), "Age", "years", [74,76,77], c(">",75))
    define(d(p,32), "PSA", "ng/mL", [0.9,1.1,1.2], c("<",1))
    reject("2e37abc94fc42dc04349c4d8b863d5d1:43",
           "Adequate individual response after AVPC is not the population percentage achieving CR; the old 92% cutoff is inapplicable.")
    p = "b8d8c75ad738f9452c3c41236516d8a5"
    define(d(p,30), "Time since treatment completion", "years", [4.9,5.1,5.2], r(3,5),
           constraints=[c(">",2)])
    define(d(p,28), "Time since treatment completion", "years", [1.9,2.1,2.2], r(0,2))
    define("78c7d9ab1aec0ac4d6c1f62a7cf0b369:2", "PET/CT Deauville score", "points", [3,4,5],
           all_of(flag("end_initial_treatment"),r(1,3)),
           [("end_initial_treatment",True,"The assessment is at the end of initial PMBL treatment.")])
    define("01de3cd3fc809c0292e2518e75bf99cb:12", "Clinical tumor diameter", "cm", [1.9,2.1,2.2],
           all_of(any_of(flag("trunk"),flag("extremity")),c(">=",2)),
           [("trunk",True,"The tumor is on the trunk."),
            ("extremity",False,"The tumor is not on an extremity.")])
    define("00af1512db52ba99bc141458324f7fc4:18", "Closest margin assessed by the treating physician", "cm",
           [0.9,1.1,1.2], c("<",1))
    p = "47453b1e7342ef125dad4c005448a426"
    reject(d(p,43), "HR-HPV infection is an individual categorical test result; a population clearance rate of 86.7% is not its threshold.")
    define(d(p,49), "产后时间", "周", [5,6,7],
           all_of(any_of(all_of(flag("postpartum"),r(6,8)),flag("post_abortion")),flag("management_concern")),
           [("postpartum",True,"患者已分娩，当前为产后。"),
            ("post_abortion",False,"没有人工流产术后状态。"),
            ("management_concern",True,"本次就诊关注阴道微生态恢复或阴道炎管理。")])
    p = "48b354078dca3fbaf7855e527e8d1637"
    define(d(p,13), "咳嗽呼气气流峰值", "L/min", [269,271,272], c("<",270))
    define(d(p,6), "用力肺活量（FVC）占预计值", "%", [49,51,52], c("<",50))
    reject("17e6529875cc41dd3eaf92792da955cb:2",
           "'Mainly seen at gestational age <35 weeks' is not a necessary-and-sufficient definition of preterm RDS; the old age-only threshold does not encode the complete condition.")
    p = "0859e1f897706e607f69626dca971398"
    define(d(p,6), "患者年龄", "岁", [17,19,20],
           any_of(all_of(flag("female"),c(">=",16)),all_of(flag("male"),c(">=",18))),
           [("female",False,"患者为男性。"),("male",True,"性别为男。")],
           constraints=[r(10,20)])
    define(d(p,2), "患者年龄", "岁", [19,21,22],
           all_of(any_of(r(10,20),flag("peripubertal_onset")),flag("stable_after_puberty")),
           [("peripubertal_onset",False,"病史记录起病在5岁，不在青春期前后。"),
            ("stable_after_puberty",True,"青春期后萎缩趋于稳定，过去2年外形无肉眼可见变化。")])
    if set(specs) != ids:
        raise ValueError("Construction pool differs from the existing 50 IDs: " + str(ids ^ set(specs)))
    by_id = {n["node_id"]: n for n in selected}
    for nid, spec in specs.items():
        node = by_id[nid]
        spec["condition"] = node["condition"]
        spec["ancestors"] = node["ancestors"]
        spec["split"] = node["split"]
        spec["source_id"] = node["source_id"]
        spec["node_mapping"] = node["node_mapping"]
        if not spec["eligible"]:
            continue
        required = fields(spec["predicate"])
        if required != set(spec["fixed_facts"]) | {"measurement"}:
            raise ValueError("Incomplete predicate facts for " + nid)
        truths = []
        for level, value in spec["values"].items():
            facts = dict(spec["fixed_facts"], measurement=value)
            if not all(evaluate(rule, facts) for rule in spec["ancestor_constraints"]):
                raise ValueError("Ancestor changes for " + nid)
            truths.append(int(evaluate(spec["predicate"], facts)))
        if truths[0] == truths[1] or truths[1] != truths[2]:
            raise ValueError("Invalid crossing/same-side pair for " + nid)
        spec["labels"] = dict(zip(spec["values"], truths))
    return list(specs.values())

def patient_for(spec, level, background):
    facts = dict(spec["fixed_facts"], measurement=spec["values"][level])
    value = facts["measurement"]
    line = f"{spec['metric']}: {value} {spec['unit']}."
    fixed = "\n".join(spec["fixed_fact_sentences"].values())
    return "\n".join(x for x in (line, fixed, spec[background+"_background"]) if x), facts

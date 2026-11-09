"""Render only the medical simulation tables from the generated summary CSV.
Run after medical_extension.py; no model or dataset evaluation is performed.
"""
from pathlib import Path
import csv
import math
import re
import runpy
from statistics import mean

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PAPER_STYLE=runpy.run_path(str(ROOT/"format_paper_tables.py"))
ROWS=list(csv.DictReader((HERE/"medical_extension_summary.csv").open(encoding="utf-8")))
MODELS=list(dict.fromkeys(r["model"] for r in ROWS))
METHODS=list(dict.fromkeys(r["method"] for r in ROWS))
NAMES={r["method"]:PAPER_STYLE["ALIASES"].get(r["method_name"],r["method_name"]) for r in ROWS}
LOOKUP={(r["task"],r["model"],r["method"]):r for r in ROWS}

def fmt(value, digits=2):
    if value is None or value=="":return "—"
    value=float(value)
    if not math.isfinite(value):return "—"
    if abs(value)<0.5*10**-digits:value=0.
    return f"{value:.{digits}f}"

def cell(row,key,sd=False):
    if key is None:return "—"
    if key=="direction_counts":
        return fmt(row["direction_degradation_mean"],1)+"/"+fmt(row["direction_eligible_n_mean"],1)
    digits=3 if key=="direction_degradation_rate" else 2
    val=fmt(row.get(key+"_mean"),digits)
    if val=="—":return val
    return val+"±"+fmt(row.get(key+"_sd"),digits) if sd else val

def table(headers,rows):
    return PAPER_STYLE["table"](headers,rows,simulation=True)

MAIN=[
 ("MedPIC",[("ER%","evidence_er"),("证据修正 RMS","evidence_correction_rms"),("方向退化%","direction_degradation_rate"),("配对集合全对%","pair_exact"),("答案集合全对%","answer_set_exact")]),
 ("MedPAIR",[("低相关删除 VR%","vr"),("关键删除响应保留%","evidence_er"),("证据修正 RMS","evidence_correction_rms"),("方向退化%","direction_degradation_rate"),("原题 Acc%","accuracy")]),
 ("MedEinst",[("ER%","evidence_er"),("证据修正 RMS","evidence_correction_rms"),("方向退化%","direction_degradation_rate"),("配对全对%","pair_exact"),("Bias Trap Rate%","bias_trap_rate")]),
 ("MedGUIDE",[("ER%",None),("方向退化%",None),("Acc%","accuracy"),("疾病宏平均%","disease_macro_accuracy")]),
 ("TrialGPT",[("ER%",None),("方向退化%",None),("平衡 Acc%","balanced_accuracy"),("Macro-F1%","macro_f1")])]
FULL=[
 ("MedPIC",[("ER%","evidence_er"),("证据修正 RMS","evidence_correction_rms"),("方向退化%","direction_degradation_rate"),("退化数/原正向数","direction_counts"),("配对集合全对%","pair_exact"),("答案集合全对%","answer_set_exact")]),
 ("MedPAIR",[("低相关删除 VR%","vr"),("低相关删除 RMS","low_delete_rms"),("低相关补回 RMS","low_addback_rms"),("关键证据删除 RMS","key_delete_rms"),("关键证据补回 RMS","key_addback_rms"),("关键删除响应保留%","evidence_er"),("证据修正 RMS","evidence_correction_rms"),("方向退化%","direction_degradation_rate"),("退化数/原正向数","direction_counts"),("原题 Acc%","accuracy")]),
 ("MedPAIR",[("低相关删除 Δmargin","low_delete_signed"),("低相关补回 Δmargin","low_addback_signed"),("低相关恢复误差 RMS","low_restore_error_rms"),("关键删除 Δmargin","key_delete_signed"),("关键补回 Δmargin","key_addback_signed"),("关键恢复误差 RMS","key_restore_error_rms")]),
 ("MedEinst",[("ER%","evidence_er"),("证据修正 RMS","evidence_correction_rms"),("方向退化%","direction_degradation_rate"),("退化数/原正向数","direction_counts"),("配对全对%","pair_exact"),("Control Acc%","control_accuracy"),("Trap Acc%","trap_accuracy"),("Bias Trap Rate%","bias_trap_rate")]),
 ("MedGUIDE",[("ER%",None),("VR%",None),("方向退化%",None),("Acc%","accuracy"),("疾病宏平均%","disease_macro_accuracy")]),
 ("TrialGPT",[("ER%",None),("VR%",None),("方向退化%",None),("平衡 Acc%","balanced_accuracy"),("Macro-F1%","macro_f1")])]

def main_tables():
    tables=[]
    for task,cols in MAIN:
        rows=[]
        for method in METHODS:
            avg={}
            for _,key in cols:
                if key:
                    values=[LOOKUP[(task,model,method)].get(key+"_mean") for model in MODELS]
                    defined=[float(v) for v in values if v is not None and v!=""]
                    avg[key+"_mean"]=mean(defined) if defined else None
            rows.append([NAMES[method]]+[cell(avg,key) for _,key in cols])
        tables.append(table(["条件"]+[label for label,_ in cols],rows))
    return tables

def full_tables():
    tables=[]
    for task,cols in FULL:
        rows=[]
        for model in MODELS:
            for method in METHODS:
                row=LOOKUP[(task,model,method)]
                rows.append([model,NAMES[method]]+[cell(row,key,True) for _,key in cols])
        tables.append(table(["模型","条件"]+[label for label,_ in cols],rows))
    return tables

def replace_tables(text,start,end,tables):
    begin=text.index(start);finish=text.index(end,begin)
    section=text[begin:finish]
    pattern=r"(?m)^\|.*(?:\n\|.*)*"
    matches=list(re.finditer(pattern,section))
    assert len(matches)==len(tables),(start,len(matches),len(tables))
    for match,replacement in reversed(list(zip(matches,tables))):
        section=section[:match.start()]+replacement+section[match.end():]
    return text[:begin]+section+text[finish:]

def render():
    paper=ROOT/"paper_story_latest.zh.md"
    brief=ROOT/"meeting/meeting_brief.zh.md"
    s=paper.read_text(encoding="utf-8")
    s=replace_tables(s,"### 4.3 Medical Task Simulations","## 5 Analysis",main_tables())
    s=replace_tables(s,"### D.3 ","### D.4 ",full_tables())
    paper.write_text(s,encoding="utf-8")
    s=brief.read_text(encoding="utf-8")
    s=replace_tables(s,"### 4.3 Medical Task Simulations","## 5 Analysis",main_tables())
    brief.write_text(s,encoding="utf-8")
    PAPER_STYLE["export_acl_tables"]()

if __name__=="__main__":render()

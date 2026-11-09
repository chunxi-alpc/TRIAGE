"""Apply the paper's shared method names and metric headers without changing results.
Usage: python format_paper_tables.py
"""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parent
STYLE=json.loads((ROOT/"paper_table_style.json").read_text(encoding="utf-8"))
ALIASES=STYLE["method_aliases"]
HEADERS=STYLE["header_aliases"]
SEP=re.compile(r"^\|(?:\s*:?-+:?\s*\|)+\s*$")

def split_cells(line):
    return [x.strip() for x in re.split(r"(?<!\\)\|",line.strip().strip("|"))]

def format_header(label,simulation=False):
    label=HEADERS.get(label,label)
    if simulation:label=label.replace("(logit)","(score)")
    notation={
      "RMS_Δc,E":r"$\mathrm{RMS}_{\Delta c,E}$",
      "RMS_Δc,C":r"$\mathrm{RMS}_{\Delta c,C}$",
      "RMS_Δc,F":r"$\mathrm{RMS}_{\Delta c,F}$",
      "RMS_Δc":r"$\mathrm{RMS}_{\Delta c}$",
      "RMS_I":r"$\mathrm{RMS}_{I}$","RMS_A":r"$\mathrm{RMS}_{A}$",
      "RMS_Preserving":r"$\mathrm{RMS}_{\mathrm{pres}}$",
      "RMS_Pair":r"$\mathrm{RMS}_{\mathrm{pair}}$",
      "RMS_Base":r"$\mathrm{RMS}_{\mathrm{base}}$",
      "RMS_Edited":r"$\mathrm{RMS}_{m}$",
      "RMS_P":r"$\mathrm{RMS}_{P}$","RMS_F0":r"$\mathrm{RMS}_{F0}$",
      "VR_Preserving,Error":r"$\mathrm{VR}_{\mathrm{pres,err}}$",
      "ER_Changing,Error":r"$\mathrm{ER}_{\mathrm{chg,err}}$",
      "VR_A,Error":r"$\mathrm{VR}_{A,\mathrm{err}}$","ER_E,Error":r"$\mathrm{ER}_{E,\mathrm{err}}$",
      "VR_Low":r"$\mathrm{VR}_{\mathrm{low}}$",
      "VR_I":r"$\mathrm{VR}_{I}$","VR_A":r"$\mathrm{VR}_{A}$",
      "VR_P":r"$\mathrm{VR}_{P}$","VR_F0":r"$\mathrm{VR}_{F0}$",
      "ER_E":r"$\mathrm{ER}_{E}$","ER_C":r"$\mathrm{ER}_{C}$","ER_F":r"$\mathrm{ER}_{F}$",
      "Pair Acc._I":"Pair Acc. (I)","Pair Acc._E":"Pair Acc. (E)",
      "Pair Correct_I":"Pair Correct (I)","Pair Correct_E":"Pair Correct (E)",
      "Error-pair Correct_E":"Error-pair Correct (E)",
      "Option-pair Correct_E":"Option-pair Correct (E)",
      "R²":r"$R^2$","|Δc|":r"$\lvert\Delta c\rvert$"}
    # Exact replacement is idempotent: input terms contain no LaTeX notation.
    for old,new in notation.items():label=label.replace(old,new)
    return label

def table(headers,rows,simulation=False):
    formatted=[format_header(x,simulation) for x in headers]
    method_headers={"方法","真实测试方法","条件","模型条件","消融","对照方法","输入／配置",
                    "Method","Ablation","Input / Configuration"}
    method_indices=[j for j,h in enumerate(headers) if h in method_headers]
    # In some comparison/design tables names occur outside a dedicated Method column.
    output=[]
    for row in rows:
        output.append([ALIASES.get(cell,cell) for cell in row])
    align=["---" if i in method_indices or i==0 else "---:" for i in range(len(headers))]
    return "\n".join(["| "+" | ".join(formatted)+" |","|"+"|".join(align)+"|"]+
                     ["| "+" | ".join(row)+" |" for row in output])

def format_document(text):
    lines=text.splitlines()
    out=[];i=0;section=""
    while i<len(lines):
        if lines[i].startswith("#"):section=lines[i] if lines[i].startswith(("### 4.","## 5 ","### D.")) else section
        if i+1<len(lines) and lines[i].startswith("|") and SEP.fullmatch(lines[i+1]):
            header=split_cells(lines[i]);j=i+2;rows=[]
            while j<len(lines) and lines[j].startswith("|"):
                row=split_cells(lines[j])
                assert len(row)==len(header),(i+1,len(header),len(row))
                rows.append(row);j+=1
            simulation=section.startswith("### 4.3") or section.startswith(("### D.1","### D.2","### D.3"))
            out.extend(table(header,rows,simulation).splitlines());i=j
        else:out.append(lines[i]);i+=1
    return "\n".join(out)+"\n"

def latex_cell(value):
    # Preserve explicit math spans and escape ordinary table text.
    value=value.replace("**","")
    pieces=re.split(r"(\$[^$]*\$)",value)
    esc={"\\":r"\textbackslash{}","&":r"\&","%":r"\%","#":r"\#",
         "_":r"\_","{":r"\{","}":r"\}","~":r"\textasciitilde{}","^":r"\textasciicircum{}"}
    symbols={"↑":r"$\uparrow$","↓":r"$\downarrow$","≈":r"$\approx$",
             "Δ":r"$\Delta$","τ":r"$\tau$","δ":r"$\delta$","→":r"$\rightarrow$",
             "±":r"$\pm$","−":"-","＋":"+","－":"-","／":"/","（":" (","）":")"}
    result=[]
    for part in pieces:
        if part.startswith("$") and part.endswith("$"):result.append(part)
        else:result.append("".join(symbols.get(ch,esc.get(ch,ch)) for ch in part))
    return "".join(result)

def latex_header(label):
    if " (" in label:
        first,second=label.split(" (",1)
        return r"\shortstack{"+latex_cell(first)+r"\\ "+latex_cell("("+second)+"}"
    return latex_cell(label)

def export_acl_tables():
    """Generate booktabs/tabularx table sources for the main result tables.
    Include with the ACL style and packages booktabs, tabularx, array.
    This exports table sources only and does not assert PDF layout validation.
    """
    p=ROOT/"paper_story_latest.zh.md"
    text=p.read_text(encoding="utf-8").split("## Appendix A",1)[0]
    lines=text.splitlines();tables=[];i=0
    while i+1<len(lines):
        if lines[i].startswith("|") and SEP.fullmatch(lines[i+1]):
            heads=split_cells(lines[i]);j=i+2;rows=[]
            while j<len(lines) and lines[j].startswith("|"):
                rows.append(split_cells(lines[j]));j+=1
            tables.append((heads,rows));i=j
        else:i+=1
    assert len(tables)==20
    descriptions=STYLE["acl_captions"]
    out=["% ACL-compatible main-result table sources.",
         "% Required packages: booktabs, tabularx, array.",
         "% Figures, tables and captions must be checked in the assembled ACL PDF.",
         "% No corresponding real model runs were performed for simulated Tables 13--17.",""]
    for number,(heads,rows) in enumerate(tables,1):
        if number<3 or number>19:continue
        method_col=next((j for j,h in enumerate(heads) if h in ["Method","Ablation"]),0)
        specs=[]
        for j in range(len(heads)):
            if j==method_col:specs.append(r">{\raggedright\arraybackslash}X")
            elif j==0:specs.append(r">{\raggedright\arraybackslash}p{0.16\textwidth}")
            else:specs.append("r")
        out.extend([r"\begin{table*}[t]",r"\centering",r"\small",
                    r"\setlength{\tabcolsep}{3pt}",
                    r"\begin{tabularx}{\textwidth}{@{}"+"".join(specs)+r"@{}}",
                    r"\toprule"])
        if 13<=number<=17:
            primary=3 if number==13 or number==15 else 4 if number==14 else 2
            downstream=len(heads)-1-primary
            out.append("Method & "+r"\multicolumn{"+str(primary)+r"}{c}{Faithfulness} & "+
                       r"\multicolumn{"+str(downstream)+r"}{c}{Downstream Performance}"+r" \\")
            out.append(r"\cmidrule(lr){2-"+str(1+primary)+r"}\cmidrule(lr){"+str(2+primary)+"-"+str(len(heads))+"}")
            headvalues=[""]+heads[1:]
        else:headvalues=heads
        out.append(" & ".join(latex_header(h) for h in headvalues)+r" \\")
        out.append(r"\midrule")
        for row in rows:
            translated=[cell.replace("官方test声明（零样本迁移）","Official test (zero-shot)") for cell in row]
            out.append(" & ".join(latex_cell(c) for c in translated)+r" \\")
        out.extend([r"\bottomrule",r"\end{tabularx}",
                    r"\caption{"+descriptions[str(number)]+"}",
                    r"\label{tab:aster-main-"+str(number).zfill(2)+"}",
                    r"\end{table*}",""])
    (ROOT/"tables_acl.tex").write_text("\n".join(out),encoding="utf-8")


def main():
    for name in ["paper_story_latest.zh.md","meeting/meeting_brief.zh.md"]:
        p=ROOT/name
        p.write_text(format_document(p.read_text(encoding="utf-8")),encoding="utf-8")

if __name__=="__main__":
    main()
    export_acl_tables()

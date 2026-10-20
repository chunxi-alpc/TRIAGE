#!/usr/bin/env python3
"""Six fixed training narratives for OR and unknown exceptions; no dev cases."""
import json, hashlib
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent

def rules(kind):
    if kind=="or":
        texts=[
            "When the visitor has a valid library card or has a paid day pass, unless the library desk is closed, assign Route Cedar.",
            "When it is not the case that the visitor has a valid library card or has a paid day pass, unless the library desk is closed, assign Route Maple.",
            "When the library desk is closed, assign Route Juniper."]
    else:
        texts=[
            "When the visitor has completed registration and has requested an appointment, unless the booking office is closed, assign Route Cedar.",
            "When it is not the case that the visitor has completed registration and has requested an appointment, unless the booking office is closed, assign Route Maple.",
            "When the booking office is closed, assign Route Juniper."]
    # Preserve explicit scope of the negated whole prerequisite.
    texts[1]=texts[1].replace("it is not the case that ", "it is not the case that (").replace(", unless", "), unless")
    return [dict(rule_id=rid,text=t,action=a) for rid,t,a in zip(
        ["R_TARGET","R_ALTERNATIVE","R_EXCEPTION"],texts,["Route Cedar","Route Maple","Route Juniper"])]

def case(cid,kind,patient,states,quotes,reasons):
    return {"source_case_id":cid,"source_family_id":"new_training_"+kind,
      "input":{"question":"Which routing action follows from the visitor's current situation and all of the rules?",
               "patient":patient,"rules":rules(kind)},
      "output":[dict(rule_id=rid,state=s,evidence_quote=q,reason_short=r) for rid,s,q,r in zip(
          ["R_TARGET","R_ALTERNATIVE","R_EXCEPTION"],states,quotes,reasons)]}

def main():
    out=HERE/"data";out.mkdir(exist_ok=True)
    if (out/"examples_manifest.json").exists():raise SystemExit("Examples already frozen.")
    examples=[]
    patient="The visitor has a valid library card. The record does not say whether the visitor has a paid day pass. The library desk is open."
    examples.append(case("u_training_or_T","or",patient,["T","F","F"],
      ["The visitor has a valid library card.","The visitor has a valid library card.","The library desk is open."],
      ["A valid card establishes one OR alternative; the unknown pass does not change applicability.",
       "One OR alternative is established, so negating the entire OR is false.",
       "The desk is explicitly open, so the closed-desk exception is false."]))
    patient="The visitor does not have a valid library card. The visitor does not have a paid day pass. The library desk is open."
    examples.append(case("u_training_or_F","or",patient,["F","T","F"],
      ["The visitor does not have a valid library card. The visitor does not have a paid day pass."]*2+["The library desk is open."],
      ["Both OR alternatives are explicitly false, so the prerequisite is false.",
       "Both alternatives are explicitly false, establishing the negation of the whole OR.",
       "The desk is explicitly open, so the exception is false."]))
    patient="The record does not establish whether the visitor has a valid library card. The visitor does not have a paid day pass. The library desk is open."
    examples.append(case("u_training_or_U","or",patient,["U","U","F"],
      ["The record does not establish whether the visitor has a valid library card."]*2+["The library desk is open."],
      ["The pass is absent, but the card alternative is unresolved; the OR could still be true or false.",
       "The OR is unresolved, so its negation is also unresolved.",
       "The desk is explicitly open, so the exception is false."]))
    base="The visitor has completed registration and requested an appointment. "
    patient=base+"The booking office is open."
    examples.append(case("u_training_exception_absent","exception",patient,["T","F","F"],
      [base.strip(),base.strip(),"The booking office is open."],
      ["Both prerequisites hold and the exception is explicitly absent.",
       "Both prerequisites hold, so their combined negation is false.",
       "The office is explicitly open, so the exception is false."]))
    patient=base+"The booking office is closed."
    examples.append(case("u_training_exception_present","exception",patient,["F","F","T"],
      ["The booking office is closed.",base.strip(),"The booking office is closed."],
      ["The established closed-office exception prevents this rule from applying despite satisfied prerequisites.",
       "The negated prerequisite is false; the established exception also prevents applicability.",
       "The office is explicitly closed, establishing the exception."]))
    patient=base+"The note does not establish whether the booking office is open or closed."
    examples.append(case("u_training_exception_U","exception",patient,["U","F","U"],
      ["The note does not establish whether the booking office is open or closed.",base.strip(),
       "The note does not establish whether the booking office is open or closed."],
      ["The prerequisites hold, but the unknown exception could change applicability; absence of documentation does not establish an open office.",
       "Both prerequisites hold, making their negation false regardless of the unknown exception.",
       "The office status is unresolved, so the exception cannot be established or ruled out."]))
    obj={"count_demonstrations":2,"count_patient_narratives":6,
         "source":"New authored training examples; no evaluation patient, threshold, case ID, or gold reused.",
         "frozen_before_dev_inference":True,"demonstrations":[
             {"description":"OR true, false and unresolved; irrelevant unknown does not prevent T","cases":examples[:3]},
             {"description":"Exception explicitly absent, present and unresolved; a false prerequisite remains F","cases":examples[3:]}]}
    for c in examples:
        for r in c["output"]:assert r["evidence_quote"] in c["input"]["patient"]
    target=out/"u_examples.json"
    target.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
    (out/"examples_manifest.json").write_text(json.dumps({
      "frozen_utc":datetime.now(timezone.utc).isoformat(),"demonstrations":2,"patient_narratives":6,
      "example_file_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
      "scope":"Only examples change; original A1 system, target input and output schema remain unchanged.",
      "coverage":["OR T/F/U","NOT unresolved OR remains U","unknown exception","irrelevant unknown with decisive T or F"]},indent=2)+"\n")

if __name__=="__main__":main()

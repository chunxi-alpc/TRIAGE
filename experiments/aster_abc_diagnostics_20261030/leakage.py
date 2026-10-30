"""Read-only source overlap audit using already-recorded source membership."""
import argparse,json,itertools,subprocess
from collections import Counter,defaultdict
from pathlib import Path

def readlines(path):
    with path.open() as f: return [json.loads(line) for line in f if line.strip()]

def audit(run):
    sources=json.loads((run/"source_clusters.json").read_text())
    registry=sources["records"]
    def cluster(s): return registry.get(s,{}).get("cluster",s)
    pairs=readlines(run/"review/accepted_pairs.jsonl")
    partitions=defaultdict(set)
    for row in pairs: partitions[row["split"]].add(cluster(row["source_id"]))
    expected=["fit","calibration","selection","development","confirm"]
    overlaps=[]
    for a,b in itertools.combinations(expected,2):
        shared=sorted(partitions[a]&partitions[b])
        overlaps.append({"left":a,"right":b,"count":len(shared),"source_clusters":shared})
    official=readlines(run/"cases/official_test.jsonl")
    official_clusters={cluster(row["source_id"]) for row in official}
    learned=set().union(*(partitions[k] for k in expected))
    prior=set(sources.get("pr40_clusters",[]))
    previous=set(sources.get("previous_mechanism_clusters",[]))
    exposed=[row for row in official if cluster(row["source_id"]) in prior]
    candidates=readlines(run/"data02/candidate_pairs.jsonl")
    candidate_partitions=defaultdict(set)
    for row in candidates: candidate_partitions[row["split"]].add(cluster(row["source_id"]))
    case_partitions={}
    actual_case_sources={}
    for phase in ["calibration","development","confirm"]:
        rows=readlines(run/"cases"/(phase+".jsonl"))
        got={cluster(row["source_id"]) for row in rows}
        actual_case_sources[phase]=got
        case_partitions[phase]={"cases":len(rows),"source_clusters":len(got),
            "not_in_registered_candidate_partition":sorted(got-candidate_partitions[phase]),
            "outside_accepted_age_pairs_but_registered_for_original_case_evaluation":sorted(got-partitions[phase])}
    actual_used=set(partitions["fit"])|set(partitions["selection"])|set().union(*actual_case_sources.values())
    actual_partitions=dict(partitions)
    for phase,got in actual_case_sources.items(): actual_partitions[phase]=got
    actual_overlaps=[{"left":a,"right":b,"count":len(actual_partitions[a]&actual_partitions[b])}
        for a,b in itertools.combinations(expected,2)]

    return {
        "method":sources.get("method"),"no_new_hashes":True,
        "accepted_pair_rows":len(pairs),"accepted_unique_source_clusters":len(learned),
        "pair_rows_per_split":dict(Counter(r["split"] for r in pairs)),
        "clusters_per_split":{k:len(partitions[k]) for k in expected},
        "pairwise_partition_overlaps":overlaps,
        "registered_candidate_source_clusters":len(set().union(*candidate_partitions.values())),
        "registered_candidate_clusters_per_split":{k:len(v) for k,v in candidate_partitions.items()},
        "actual_direction_and_original_case_used_source_clusters":len(actual_used),
        "actual_used_vs_official_source_overlap":sorted(actual_used&official_clusters),
        "actual_used_vs_PR40_exposure_overlap":sorted(actual_used&prior),
        "actual_used_vs_previous_mechanism_overlap":sorted(actual_used&previous),
        "actual_used_partition_overlaps":actual_overlaps,
        "partition_note":"Original-case evaluation uses registered candidate source partitions, including sources rejected for age-triple consistency; these are not missing from the evaluation contract.",

        "learned_vs_official_source_overlap":sorted(learned&official_clusters),
        "learned_vs_PR40_exposed_overlap":sorted(learned&prior),
        "learned_vs_previous_mechanism_overlap":sorted(learned&previous),
        "clinical_case_partition_consistency":case_partitions,
        "official_planned_cases":len(official),"official_source_clusters":len(official_clusters),
        "official_prior_PR40_exposed_cases":len(exposed),
        "official_prior_PR40_exposed_clusters":len({cluster(row["source_id"]) for row in exposed}),
        "official_prior_exposure_interpretation":"Preserve the official full denominator. Report exposure-stratified descriptions separately; do not claim every official source is wholly unexposed.",
        "clinical_scope":"Current-age deterministic predicates; zero physician-adjudicated clinical triples.",
        "remaining_uncertainty":"Operational content/title/DOI clustering does not rule out all cross-language/version overlap. MedGUIDE external independence remains unverified.",
        "frozen_files_modified":False
    }

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    result=audit(a.run)
    (a.out/"source_leakage_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"source_clusters":result["accepted_unique_source_clusters"],"learning_official_overlap":len(result["learned_vs_official_source_overlap"]),"official_PR40_exposed_cases":result["official_prior_PR40_exposed_cases"]}))
if __name__=="__main__": main()

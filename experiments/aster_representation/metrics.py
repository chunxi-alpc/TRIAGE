import re

def strict_parse(text):
    leaves = re.findall('FINAL\\s*[:：]\\s*(L\\d+|NONE)\\b', text, re.I)
    sections = re.findall('PATH\\s*[:：]\\s*([^\\n]*)', text, re.I)
    path = re.findall('R\\d+[YN]', sections[-1]) if sections else []
    leaf = leaves[-1].upper() if leaves else None
    nodes = [rid[:-1] for rid in path]
    invalid = leaf is None or not sections or len(set(nodes)) != len(nodes)
    return (leaf, path, invalid)

def strict_score(case, text):
    leaf, path, bad = strict_parse(text)
    return {'leaf_correct': leaf == case['gold_leaf'], 'root_correct': bool(path) and path[0] == case['gold_rules'][0], 'exact_path': not bad and path == case['gold_rules'], 'invalid': bad, 'leaf': leaf, 'path': path}

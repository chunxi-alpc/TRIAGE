import re

def tree_guideline(tree):
    nodes = []

    def visit(n, parent=None, branch=None):
        i = len(nodes)
        nodes.append((n, parent, branch))
        if any((isinstance(n.get(b), dict) for b in ['yes', 'no'])):
            if not all((isinstance(n.get(b), dict) for b in ['yes', 'no'])):
                raise ValueError('nonbinary tree')
            visit(n['yes'], i, 'Y')
            visit(n['no'], i, 'N')
    visit(tree)
    loc = {id(n): i for i, (n, _, _) in enumerate(nodes)}
    groups = []
    mapping = {}
    for i, (n, parent, branch) in enumerate(nodes):
        original = str(n['index'])
        mapping[original] = {'node': i, 'leaf': not any((isinstance(n.get(b), dict) for b in ['yes', 'no'])), 'description': n.get('conclusion', n.get('description', ''))}
        if not any((isinstance(n.get(b), dict) for b in ['yes', 'no'])):
            continue
        ancestors = []
        cur = i
        while nodes[cur][1] is not None:
            pi = nodes[cur][1]
            br = nodes[cur][2]
            pn = nodes[pi][0]
            ancestors.append(pn['description'] + ('（判断为是）' if br == 'Y' else '（判断为否）'))
            cur = pi
        group = {'group_id': f'G{i}', 'mutually_exclusive': True, 'ancestor_conditions': list(reversed(ancestors)), 'rules': []}
        if parent is not None:
            group.update(parent_group_id=f'G{parent}', parent_rule_id=f'R{parent}{branch}')
        for b, B in [('yes', 'Y'), ('no', 'N')]:
            child = n[b]
            j = loc[id(child)]
            leaf = not any((isinstance(child.get(k), dict) for k in ['yes', 'no']))
            action = f'L{j}：' + child.get('conclusion', child.get('description', '')) if leaf else f'继续评估规则组 G{j}。'
            group['rules'].append({'rule_id': f'R{i}{B}', 'condition': n['description'] + ('（判断为是）' if B == 'Y' else '（判断为否）'), 'action': action})
        groups.append(group)
    if not groups:
        raise ValueError('no decision groups')
    return ({'groups': groups}, mapping)
AGE = re.compile('(?P<num>\\d{1,3}(?:\\.\\d+)?)\\s*(?:(?:[-– ]?(?:years?|yrs?)[-– ]old)|(?:周岁|岁))', re.I)

def clean_anchor(patient):
    matches = list(AGE.finditer(patient))
    if not matches or matches[0].start() > 140:
        raise ValueError('no explicit current-age anchor near start')
    vals = {float(m.group('num')) for m in matches}
    if len(vals) != 1:
        raise ValueError('multiple age values or family/history age references')
    age = vals.pop()
    if age < 0:
        raise ValueError('negative age')
    body = AGE.sub('', patient)
    body = re.sub('\\b(elderly|adult|adolescent|pediatric|paediatric|young|middle-aged|older|aged)\\b', '', body, flags=re.I)
    body = re.sub('\\b(child|boy|girl|infant|baby|teenager)\\b', 'patient', body, flags=re.I)
    body = re.sub('老年|中年|青年|成年|儿童|患儿|婴幼儿|婴儿', '患者', body)
    body = re.sub('\\bAn (?=man|woman|male|female|patient)', 'A ', body, flags=re.I)
    body = re.sub('[,，]{2,}', '，', body)
    body = re.sub('[ \\t]+', ' ', body).strip()
    return (age, body)

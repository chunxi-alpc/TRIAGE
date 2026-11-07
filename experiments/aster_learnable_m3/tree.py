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

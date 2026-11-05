"""Keep parenthetical definitions and citation markers in conservative deletions."""
import re

def preserve_notes(cuts, blocks):
    revised, retained = [], []
    pairs={"(":")","（":"）","[":"]","【":"】"}
    for cut in cuts:
        quote=cut["quote"]
        stack=[];protected=[];start=None
        for i,char in enumerate(quote):
            if char in pairs:
                if not stack:
                    start=i
                stack.append(pairs[char])
            elif stack and char==stack[-1]:
                stack.pop()
                if not stack:
                    protected.append((start,i+1))
        for match in re.finditer(r"\$\s*\^\{[^}]+\}\s*\$|\^\{[^}]+\}",quote):
            protected.append(match.span())
        merged=[]
        for a,b in sorted(protected):
            if merged and a<=merged[-1][1]:
                merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
            else:
                merged.append((a,b))
        cursor=0
        for a,b in merged+[(len(quote),len(quote))]:
            fragment=quote[cursor:a].strip()
            if fragment and re.search(r"\w",fragment):
                revised.append(dict(block_id=cut["block_id"],quote=fragment))
            if b>a:
                retained.append(dict(block_id=cut["block_id"],quote=quote[a:b],
                    reason="Parenthetical explanation or citation retained conservatively"))
            cursor=b
    return revised,retained

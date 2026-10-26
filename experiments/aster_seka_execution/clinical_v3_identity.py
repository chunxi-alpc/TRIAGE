"""Recover derived sample IDs from the frozen identity table without changing records."""
def index_predictions(rows, frozen):
    keys = ('id', 'case_path_id', 'path_index', 'filename')
    lookup = {tuple(entry['identity'][key] for key in keys): entry['sample_id']
              for entry in frozen['selected']}
    if len(lookup) != len(frozen['selected']):
        raise ValueError('Duplicate frozen identities')
    indexed = {}
    for row in rows:
        identity = tuple(row[key] for key in keys)
        if identity not in lookup:
            raise ValueError('Prediction outside frozen development set')
        sid = lookup[identity]
        if row.get('sample_id', sid) != sid or sid in indexed:
            raise ValueError('Conflicting or duplicate prediction identity')
        indexed[sid] = row
    return indexed

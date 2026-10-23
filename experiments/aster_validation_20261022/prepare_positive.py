"""Apply upstream PASTA CounterFact formatting, recording code/data fingerprints."""
import ast,functools,hashlib,json,logging,sys,urllib.request
from pathlib import Path
import spacy
root=Path(sys.argv[1]);source=root/'pasta_data.py'
url='https://raw.githubusercontent.com/QingruZhang/PASTA/main/evaluation/data.py'
if not source.exists():
 with urllib.request.urlopen(url,timeout=60) as r:source.write_bytes(r.read())
names={'_COUNTERFACT_PARAPHRASE_PROMPT_ARTIFACTS','_rejoin_sents_on_entity','_strip_counterfact_paraphrase_prompt','_reformat_counterfact_sample'}
nodes=[]
for node in ast.parse(source.read_text()).body:
 if isinstance(node,ast.FunctionDef) and node.name in names:nodes.append(node)
 elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):nodes.append(node)
ns={'ContextMediationSample':dict,'load_spacy_model':functools.cache(spacy.load),'logger':logging.getLogger(__name__)}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
# Require the real parser. Do not silently replace it with a blank model.
spacy.load('en_core_web_sm')
raw=root/'counterfact_original.json';rows=json.loads(raw.read_text());out=root/'counterfact_prepared.jsonl'
with out.open('w') as f:
 for row in rows:f.write(json.dumps(ns['_reformat_counterfact_sample'](row))+'\n')
(root/'positive_preparation.json').write_text(json.dumps({'n':len(rows),'source_url':'https://rome.baulab.info/data/dsets/counterfact.json','raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'formatter_url':url,'formatter_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'prepared_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'spacy':spacy.__version__,'en_core_web_sm':spacy.load('en_core_web_sm').meta['version'],'deviation':'Original source + upstream PASTA formatter; SEKA release byte identity unverified.'},indent=2))

folder=root/'counterfact_official';folder.mkdir(exist_ok=True)
link=folder/'counterfact.jsonl'
if not link.exists():link.symlink_to(out)

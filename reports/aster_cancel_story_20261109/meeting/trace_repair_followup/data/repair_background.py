"""Original repair-v2 directly located numeric-background edit, verbatim."""
import re
from experiments.aster_closed_form.data import condition
NUM=r"-?\d+(?:\.\d+)?"
def background_from_measurements(g):
 context=condition(g)+'\n'+'\n'.join(a['condition'] for a in g['ancestors'])
 families=[('收缩压',['收缩压','血压','循环','血流动力','低血压','高血压'],'mmHg'),('心率',['心率','心动','心律','脉搏','循环','血流动力'],'次/分'),('呼吸频率',['呼吸','气促','气短','喘','RR','肺'],'次/分'),('体温',['体温','发热','感染','热'],'℃'),('BMI',['BMI','体重','肥胖','超重','营养'],'kg/m²')]
 for field,aliases,unit in families:
  if any(a in context for a in aliases):continue
  pattern=re.escape(field)+r'(?:为|测得为|测得|约为|约|是|\s)*\s*('+NUM+r')\s*'+re.escape(unit)
  ms=list(re.finditer(pattern,g['patient']))
  if len(ms)!=1:continue
  m=ms[0];v=float(m[1]);step=.1 if '.' in m[1] else 1;new=v+step
  # Stay on the same side of all explicit comparisons of that measurement in the case.
  if any((v<t)!=(new<t) or (v<=t)!=(new<=t) for t in [float(x[1]) for x in re.finditer(r'(?:[<>≥≤]|低于|高于|超过|达到)\s*('+NUM+r')\s*'+re.escape(unit),g['patient'])]):continue
  old=m.group();updated=old[:m.start(1)-m.start()]+format(new,'.12g')+old[m.end(1)-m.start():]
  return dict(old=old,new=updated,field=field,before=m[1],after=format(new,'.12g'),reason='Located background measurement; variable and related terms absent from current and ancestor conditions; reported comparison sides preserved',unrelated_to_all_conditions=True,method='program-background')
 return None

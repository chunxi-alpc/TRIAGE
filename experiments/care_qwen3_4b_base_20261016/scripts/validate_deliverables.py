import pathlib,json
from PIL import Image
import pandas as pd
root=pathlib.Path('.')
names=['O1_faithfulness_misuse','O2_pca_pairwise','O2_layer_head_heatmaps','O2_controls_and_probes',
       'O3_z_histogram','O3_weight_delta_scatter','O3_mismatch_heatmap']
items=[]
for name in names:
    for ext in ['png','pdf','svg']:
        p=root/'figures'/f'{name}.{ext}'
        assert p.is_file() and p.stat().st_size>1000,p
        if ext=='png':
            im=Image.open(p); assert min(im.info['dpi'])>=299
            assert min(im.size)>700
        items.append({'path':str(p),'bytes':p.stat().st_size})
df=pd.read_csv(root/'results/obs1/predictions.csv')
assert not df.prior_eligible.any()
assert not any('layout_ONLY' in x['path'] for x in items)
(root/'results/checks/deliverables.json').write_text(json.dumps({'paper_figures':len(names),'files':items,'dpi':300},indent=2))
print(f'Validated {len(names)} scientific figures × PNG/PDF/SVG; no template images counted.')

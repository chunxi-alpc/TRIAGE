"""CPU-only audit and anonymous per-example export of completed development runs."""
import csv,gzip,hashlib,json
import numpy as np
from reproduce import ROOT,prepare,metric
from balanced_analysis import prediction,paired

def main():
    result=json.loads((ROOT/'balanced_fresh.json').read_text())
    execution=json.loads((ROOT/'execution.json').read_text())
    report=['# ASTER 完成核对与分组结果','','两模型各800条真实前向完成，以下全部来自同一seed=0开发集；没有新增GPU运行。','',
    '## 主要结论','','同参数移除残差后，Qwen准确率63.125%，Llama 62.875%；加入残差后均为63.250%，分别仅净增1条和3条。两者按树配对区间均跨零。动态融合分别为63.000%和62.875%。目前不能把总体提升归因于注意力残差，也没有证据证明稳定优势。', '',
    '退化/纠正均相对原模型：原答对变错/原答错变对。表内分母为对应变体200条；这不是以原答对数为分母的条件退化率。四个变体属于相同200病例，不是800个独立病例。','',
    '## 各变体结果','','|模型|方法|NC准确率|IC准确率|AC准确率|SW准确率|NC退化数|IC退化数|AC纠正数|SW纠正数|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    audit={}; exports=[]
    names={'original':'原模型','static':'静态融合','dynamic':'动态融合','no_delta':'同参数无残差','residual':'注意力残差','numeric_null':'同参数零强度数值对照','option_rotation':'同参数错位','sign_flip':'同参数符号翻转','reverse':'同参数反向残差'}
    for model in ('qwen','llama'):
        with gzip.open(ROOT/'fresh_numeric'/f'{model}.jsonl.gz','rt') as f: raw=[json.loads(s) for s in f]
        oldfile=ROOT/'data'/f"{model}_dev_seed0_{'logits_numeric' if model=='qwen' else 'numeric'}.jsonl"
        old=[json.loads(s) for s in oldfile.read_text().splitlines()]
        key=lambda r:(r['id'],r['variant'],r['seed'])
        assert len(raw)==800 and len({key(r) for r in raw})==800
        assert {key(r):r['gold_option'] for r in raw}=={key(r):r['gold_option'] for r in old}
        assert all(r['split']=='dev' and r['seed']==0 for r in raw)
        required={'original','zero_all','zero_last_half','action_bias_1.0_all','action_bias_2.0_all','action_bias_2.0_last_half','mass_zero_control','mass_0.5_all','mass_0.5_last_half','mass_1.0_last_half'}
        for r in raw:
            assert required<=r['scores'].keys()
            for k,v in r['scores'].items():
                assert len(v)==len(r['option_S']) and np.isfinite(v).all() and int(np.argmax(v))==r['answers'][k]
        rows=[prepare(r) for r in raw]; x=result['models'][model]
        preds={'original':[r['answers']['original'] for r in rows]}
        for k in ('static','dynamic','residual'):
            preds[k]=[prediction(r,x[k]['folds'][r['tree']]['parameters']) for r in rows]
            assert preds[k]==x[k]['predictions']
        for k in ('no_delta','numeric_null','option_rotation','sign_flip','reverse'):
            preds[k]=[prediction(r,{**x['residual']['folds'][r['tree']]['parameters'],'kind':k}) for r in rows]
        assert preds['no_delta']==x['residual']['same_parameters_without_residual']
        audit[model]={'rows':800,'cases':len({r['id'] for r in rows}),'verified':True,'methods':{},'residual_vs_no_delta':paired(rows,preds['residual'],preds['no_delta']),'residual_vs_dynamic':paired(rows,preds['residual'],preds['dynamic'])}
        for k,pp in preds.items():
            z=metric(rows,pp); vv={}
            for v in ('NC','IC','AC','SW'):
                ix=[i for i,r in enumerate(rows) if r['variant']==v]
                assert len(ix)==200
                vv[v]={'n':len(ix),'correct':sum(pp[i]==rows[i]['gold_option'] for i in ix),'regressed':sum(preds['original'][i]==rows[i]['gold_option'] and pp[i]!=rows[i]['gold_option'] for i in ix),'corrected':sum(preds['original'][i]!=rows[i]['gold_option'] and pp[i]==rows[i]['gold_option'] for i in ix)}
            audit[model]['methods'][k]={'metrics':z,'by_variant_counts':vv}
            acc='|'.join(f"{vv[v]['correct']/2:.1f}%" for v in ('NC','IC','AC','SW'))
            report.append(f"|{model}|{names[k]}|{acc}|{vv['NC']['regressed']}|{vv['IC']['regressed']}|{vv['AC']['corrected']}|{vv['SW']['corrected']}|")
        for i,r in enumerate(rows):
            exports.append({'model':model,'case_id':hashlib.sha256(r['id'].encode()).hexdigest()[:16],'tree':r['tree'],'variant':r['variant'],'seed':r['seed'],'gold_option':r['gold_option'],**{k:v[i] for k,v in preds.items()}})
    report+=['','融合的主要收益来自AC/SW。Llama IC原模型68.0%，静态60.0%、动态62.5%、残差64.0%，仍存在退化。Qwen同参数加入残差后NC从76.5%降至75.5%，原答对变错由4条增至11条；Llama对应NC退化由9条增至12条。Llama同参数符号翻转为63.375%，还高于正常残差63.250%，不能把残差结果解释为明确的方向特异性证据。']
    report+=['','## 同参数负对照','','全部沿用注意力残差各折选择的融合参数，不重新选参。表内为整体准确率。','','|模型|无残差|残差|数值对照|错位|符号翻转|反向残差|','|---|---:|---:|---:|---:|---:|---:|']
    for model,a in audit.items():
        report.append('|'+model+'|'+'|'.join(f"{100*a['methods'][k]['metrics']['accuracy']:.3f}%" for k in ('no_delta','residual','numeric_null','option_rotation','sign_flip','reverse'))+'|')
    report+=['','## 逐树配对：残差对同参数无残差','','|模型|树|纠正|退化|净变化（百分点）|','|---|---|---:|---:|---:|']
    for model,a in audit.items():
        for tree,z in a['residual_vs_no_delta']['by_tree'].items():
            report.append(f"|{model}|{tree}|{z['corrected_vs_comparator']}|{z['regressed_vs_comparator']}|{100*z['accuracy_delta']:+.3f}|")
    report+=['','## 资源、复现与未完成项','','实测Qwen 60.16分钟、15.77 GiB峰值分配显存；Llama 80.64分钟、15.47 GiB。合计约2.347 GPU小时；并行运行墙钟不可按两者相加。未调用付费推理API，服务器计费单价不可得，货币成本未知。','',
    '完整主报告见 [fresh_report.zh.md](fresh_report.zh.md)，原始匿名数值见 [fresh_numeric](fresh_numeric)，逐例各方法答案见 [completed_predictions.csv](completed_predictions.csv)，分组和配对统计见 [completed_audit.json](completed_audit.json)。全部原始负结果保留。','',
    '在gcx261014环境的仓库根目录执行 `CUDA_VISIBLE_DEVICES="" python experiments/aster_dynamic_fusion_seed0_20261025/audit_completed.py` 即可从仓库压缩匿名结果复核本报告；无需模型权重、GPU或私有病例原文。完整选参复算步骤见README。','',
    '未完成独立新数据验证：已有资源55棵树中53棵已被使用，剩余2棵无病例，没有可直接运行的独立新增标注集。不能以旧P3数据代替独立确认，也不能声称注意力机制或临床有效性已得到验证。']
    (ROOT/'completed_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    with (ROOT/'completed_predictions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(exports[0]));writer.writeheader();writer.writerows(exports)
    (ROOT/'COMPLETED.zh.md').write_text('\n'.join(report)+'\n')
    print({m:{k:round(z['metrics']['accuracy']*100,3) for k,z in a['methods'].items()} for m,a in audit.items()})
if __name__=='__main__': main()

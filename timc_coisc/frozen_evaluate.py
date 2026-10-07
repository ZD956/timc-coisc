"""Reviewed COISC frozen statistics over explicitly supplied in-memory records.

NO dataset/checkpoint/manifest reader. All real-reader entry points fail closed,
even when passed authorized=True. This module neither trains nor loads models.
Membership is an in-memory list supplied by a future separately authorized I/O
layer. In this phase ONLY artificial unit fixtures may be supplied.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import numpy as np

SEEDS=(20260922,20260923,20260924)
POINTS=('T1','T2','T3','T4','T5','T6','Tp')
HORIZONS=(10,30,60,90,120)
LAGS=np.array([102.108,77.091,30.108,50.112,10.0535,180.,39.104])
PRE=np.arange(1,13)[:,None]*10 < np.maximum(0,LAGS-10)
POST=np.arange(1,13)[:,None]*10 > LAGS+10
REPLICATES=20000
BOOTSTRAP_SEED=20260926
EPS_SIGN=.0001

def read_locked_membership(*args,**kwargs):
    raise PermissionError('Membership I/O disabled: this candidate is not authorized to unseal')

def read_frozen_source(*args,**kwargs):
    raise PermissionError('Numeric source I/O disabled: this candidate is not authorized to unseal')

def run_frozen_evaluation(*args,**kwargs):
    raise PermissionError('Real frozen evaluation disabled; in-memory fixtures only in this phase')

def finite(a):
    x=np.asarray(a,dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite numeric input or statistic')
    return x

class SourceMedian:
    """Exact source mass; binary CDF search; integer check at half-mass ties.

    Sorted per-source samples are retained. No mean-of-medians approximation.
    The small floating guard below only dispatches ambiguous CDF comparisons to
    exact integer arithmetic; it never widens a statistical gate/tolerance.
    """
    def __init__(self,cells):
        self.cells=[np.sort(finite(c).ravel()) for c in cells]
        self.sizes=np.array([len(c) for c in self.cells],dtype=np.int64)
        self.empty=not np.any(self.sizes)
        if self.empty:
            self.values=np.array([]);return
        if np.any(self.sizes==0):
            raise ValueError('mixed missing PSM cells; do not drop a source')
        self.values=np.unique(np.concatenate(self.cells))
        self.lcm=math.lcm(*map(int,self.sizes))

    def many(self,counts):
        counts=np.asarray(counts)
        if counts.ndim!=2 or counts.shape[1]!=len(self.cells):
            raise ValueError('source multiplicity shape mismatch')
        if not np.isfinite(counts).all() or np.any(counts<0) or np.any(counts!=np.floor(counts)) or np.any(counts.sum(1)==0):
            raise ValueError('invalid source multiplicities')
        if self.empty:
            return None
        counts=counts.astype(np.int64)
        answer=np.empty(len(counts),np.float64)
        for start in range(0,len(counts),1024):
            cc=counts[start:start+1024]; total=cc.sum(1)
            lo=np.zeros(len(cc),dtype=np.int64); hi=np.full(len(cc),len(self.values)-1,dtype=np.int64)
            while np.any(lo<hi):
                mid=(lo+hi)//2
                cdf=sum(cc[:,i]*np.searchsorted(c,self.values[mid],side='right')/len(c) for i,c in enumerate(self.cells))
                left=cdf>=total/2
                hi=np.where(left,mid,hi);lo=np.where(left,lo,mid+1)
            answer[start:start+len(cc)]=self.values[lo]
            at=sum(cc[:,i]*np.searchsorted(c,self.values[lo],side='right')/len(c) for i,c in enumerate(self.cells))
            before_values=self.values[np.maximum(lo-1,0)]
            prev=sum(cc[:,i]*np.searchsorted(c,before_values,side='right')/len(c) for i,c in enumerate(self.cells))
            # Only near-half CDF boundaries need the exact integer tie path.
            ambiguous=(np.abs(at-total/2)<1e-10)|((lo>0)&(np.abs(prev-total/2)<1e-10))
            for j in np.flatnonzero(ambiguous):
                c=cc[j];index=lo[j]
                value=self.values[index]
                cumulative=sum(int(c[i])*int(np.searchsorted(src,value,side='right'))*(self.lcm//len(src)) for i,src in enumerate(self.cells))
                twice=2*cumulative; mass=int(c.sum())*self.lcm
                # Correct the candidate using exact rational CDF comparisons.
                if twice<mass:
                    value=min(src[k] for i,src in enumerate(self.cells) if c[i]>0 and (k:=int(np.searchsorted(src,value,side='right')))<len(src))
                else:
                    previous=[src[k-1] for i,src in enumerate(self.cells) if c[i]>0 and (k:=int(np.searchsorted(src,value,side='left')))>0]
                    if previous:
                        pv=max(previous)
                        pm=2*sum(int(c[i])*int(np.searchsorted(src,pv,side='right'))*(self.lcm//len(src)) for i,src in enumerate(self.cells))
                        if pm>=mass: value=pv
                cum=2*sum(int(c[i])*int(np.searchsorted(src,value,side='right'))*(self.lcm//len(src)) for i,src in enumerate(self.cells))
                if cum==mass:
                    nxt=min(src[k] for i,src in enumerate(self.cells) if c[i]>0 and (k:=int(np.searchsorted(src,value,side='right')))<len(src))
                    value=(value+nxt)/2
                answer[start+j]=value
        return answer

def weighted_psm(cells,multiplicities=None):
    estimator=SourceMedian(cells)
    counts=np.ones(len(cells),dtype=int) if multiplicities is None else np.asarray(multiplicities)
    result=estimator.many(counts[None,:])
    return None if result is None else float(result[0])

def membership_order(membership):
    if len(membership)!=29: raise ValueError('expected all29 locked sources')
    ids=[r['source_id'] for r in membership]
    if len(set(ids))!=29 or any(not isinstance(s,str) or not s for s in ids):
        raise ValueError('duplicate or invalid source identity')
    ext=sorted(r['source_id'] for r in membership if r['role']=='frozen_external')
    final=sorted(r['source_id'] for r in membership if r['role']=='frozen_final')
    if len(ext)!=23 or len(final)!=6: raise ValueError('locked23/6 roles/counts mismatch')
    return ext+final

def make_draw_table(membership):
    order=membership_order(membership)
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    # Fixed RNG consumption: full external matrix first, then full final matrix.
    indices=np.c_[rng.integers(0,23,size=(REPLICATES,23)),rng.integers(23,29,size=(REPLICATES,6))].astype('<i8')
    counts=np.zeros((REPLICATES,29),dtype=np.int64)
    np.add.at(counts,(np.repeat(np.arange(REPLICATES),29),indices.ravel()),1)
    digest=hashlib.sha256(indices.tobytes(order='C')).hexdigest()
    return dict(source_order=order,indices=indices,counts=counts,sha256=digest,seed=BOOTSTRAP_SEED)

def draw_means(values,counts):
    values=finite(values);counts=finite(counts)
    if counts.ndim!=2 or len(values)!=counts.shape[1] or np.any(counts.sum(1)<=0):
        raise ValueError('invalid cluster draw shape/mass')
    return np.tensordot(counts/counts.sum(1,keepdims=True),values,axes=(1,0))

def percentile_ci(values):
    x=finite(values)
    if len(x)<2: raise ValueError('insufficient bootstrap replicates')
    return np.percentile(x,[2.5,97.5],method='linear').tolist()

def effect_row(r0,coisc,b0,b1):
    finite([r0,coisc]); b0=finite(b0);b1=finite(b1)
    if b0.shape!=b1.shape: raise ValueError('unpaired bootstrap estimates')
    undefined=int((b0==0).sum())
    return dict(R0=float(r0),COISC=float(coisc),absolute_change=float(coisc-r0),
                absolute_ci=percentile_ci(b1-b0),relative_change_percent=float(100*(coisc/r0-1)) if r0!=0 else None,
                relative_ci=percentile_ci(100*(b1/b0-1)) if not undefined else None,
                undefined_relative_draws=undefined,n_replicates=len(b0),status='OK')

def decide_gates(data):
    try:
        for metric,row in data.items():
            for v in row.values(): finite(v)
            for k,v in row.items():
                if k.endswith('_ci') and (len(v)!=2 or v[0]>v[1]): raise ValueError('invalid CI')
        m,p,w,s=(data[x] for x in ('MAE','PLE','WSR','PSM'))
        gates={
            'Tp_120s_MAE':dict(value=m['COISC']-m['R0'],threshold=.005,ci=m['contrast_ci'],
                               decision='PASS' if m['contrast_ci'][1]<=.005 else 'FAIL'),
        }
        for name,row in (('PLE',p),('WSR',w)):
            ok=row['R0']>0 and row['COISC']<=.9*row['R0'] and row['contrast_ci'][1]<=0
            gates[name]=dict(R0=row['R0'],value=row['COISC'],threshold=.9*row['R0'],ci_of_COISC_minus_09R0=row['contrast_ci'],ci_threshold=0.,decision='PASS' if ok else 'FAIL')
        ok=s['COISC']>=max(.2*s['R0'],.0005) and s['ratio_contrast_ci'][0]>=0 and s['floor_contrast_ci'][0]>=0
        gates['PSM']=dict(R0=s['R0'],value=s['COISC'],threshold=max(.2*s['R0'],.0005),ratio_contrast_ci=s['ratio_contrast_ci'],floor_contrast_ci=s['floor_contrast_ci'],ci_threshold=0.,decision='PASS' if ok else 'FAIL')
        return dict(decision='PASS' if all(x['decision']=='PASS' for x in gates.values()) else 'FAIL',gates=gates)
    except (ValueError,KeyError,TypeError,IndexError) as e:
        return dict(decision='NOT_EVALUABLE',reason=str(e),gates={name:{'decision':'NOT_EVALUABLE'} for name in ('Tp_120s_MAE','PLE','WSR','PSM')})

def overall_decision(per_seed):
    if set(per_seed)!=set(SEEDS): return 'NOT_EVALUABLE'
    states=[per_seed[s]['decision'] for s in SEEDS]
    if 'NOT_EVALUABLE' in states: return 'NOT_EVALUABLE'
    return 'PASS' if all(s=='PASS' for s in states) else 'FAIL'

def validate_pairs(pairs,membership):
    order=membership_order(membership)
    if set(pairs)!=set(SEEDS): raise ValueError('missing/extra model seed')
    shared=('window_ids','target','origin','history','future')
    for seed in SEEDS:
        if set(pairs[seed])!=set(order): raise ValueError('missing/extra source; no subset analysis')
        for source in order:
            pair=pairs[seed][source]
            if set(pair)!=set(('R0','COISC')): raise ValueError('missing paired model')
            for name in ('R0','COISC'):
                r=pair[name]; n=len(r['window_ids'])
                if n==0 or len(set(np.asarray(r['window_ids']).tolist()))!=n: raise ValueError('empty/duplicate window IDs')
                for key,shape in (('prediction',(n,12,7)),('target',(n,12,7)),('sensitivity',(n,12,7)),('history',(n,20,12)),('future',(n,12,6)),('origin',(n,7))):
                    if finite(r[key]).shape!=shape: raise ValueError(f'invalid {key} shape')
                for key in shared:
                    if not np.array_equal(r[key],pairs[SEEDS[0]][source]['R0'][key]):
                        raise ValueError(f'paired/cross-seed {key} mismatch')
    return order

def source_summary(prediction,target,sensitivity):
    error=finite(prediction)-finite(target); sens=finite(sensitivity)
    if error.ndim!=3 or error.shape[1:]!=(12,7) or sens.shape!=error.shape or len(error)==0:
        raise ValueError('source array shape mismatch')
    selected=error[:,np.array(HORIZONS)//10-1]
    with np.errstate(over='raise',invalid='raise'):
        squared=selected**2
    pre=[sens[:,PRE[:,o],o].ravel() for o in range(7)]
    post=[sens[:,POST[:,o],o].ravel() for o in range(7)]
    return dict(n=len(error),MAE=np.abs(selected).mean(0),MSE=squared.mean(0),bias=selected.mean(0),
                sum_abs_error=np.abs(selected).sum(0),sum_squared_error=squared.sum(0),sum_error=selected.sum(0),
                PLE=[float(np.abs(x).mean()) if len(x) else None for x in pre],
                WSR=[float((x < -EPS_SIGN).mean()) if len(x) else None for x in post],
                PSM=[float(np.median(np.abs(x))) if len(x) else None for x in post],
                pre_count=[len(x) for x in pre],post_count=[len(x) for x in post],
                post_cells=[np.abs(x) for x in post],sensitivity=sens)

def _prediction_rows(seed,variant,cohort,source_id,aggregation,mae,mse,bias,n_sources,n_windows):
    rows=[];rmse=np.sqrt(mse)
    for hi,h in enumerate(HORIZONS):
        for oi,point in enumerate(POINTS+('all7',)):
            at=lambda a:float(a[hi,oi]) if oi<7 else float(a[hi].mean())
            rows.append(dict(seed=seed,variant=variant,cohort=cohort,source_id=source_id,aggregation=aggregation,horizon_s=h,output=point,
                             n_sources=n_sources,n_windows=n_windows,MAE_C=at(mae),RMSE_C=at(rmse),bias_C=at(bias),
                             pooled_RMSE_C=float(np.sqrt(mse[hi].mean())) if oi==7 else None))
    return rows

def evaluate(pairs,membership,diagnostics=None):
    """Full paired statistics; accepts arrays only, returns results without I/O.

    A missing/invalid source invalidates the complete confirmatory comparison;
    raw supplied records and failure reason remain in the returned bundle.
    """
    try:
        order=validate_pairs(pairs,membership)
        return _evaluate_valid(pairs,membership,order,diagnostics)
    except (ValueError,KeyError,TypeError,IndexError,FloatingPointError,OverflowError) as exc:
        invalid=[dict(seed=s,decision='NOT_EVALUABLE',gates={k:dict(decision='NOT_EVALUABLE',value=None,ci=None,reason=str(exc)) for k in ('Tp_120s_MAE','PLE','WSR','PSM')}) for s in SEEDS]
        return dict(decision=dict(overall_decision='NOT_EVALUABLE',overall_PASS=False,reason=str(exc),
                    per_seed=invalid),raw_invalid_input=pairs,
                    source_audit=[],prediction=[],sensitivity=[],bootstrap=[],collateral=[],per_seed=invalid,raw=[],
                    diagnostics=diagnostics or {},status='NOT_EVALUABLE')

def _evaluate_valid(pairs,membership,order,diagnostics):
    draw=make_draw_table(membership)
    roles={r['source_id']:r['role'] for r in membership}
    result=dict(draw_table=draw,prediction=[],sensitivity=[],sensitivity_distribution=[],sufficient_statistics=[],source_audit=[],collateral=[],bootstrap=[],per_seed=[],raw=[],diagnostics=diagnostics or {})
    decisions={}
    cohorts={'all29':np.arange(29),'frozen_external':np.arange(23),'frozen_final':np.arange(23,29)}
    for seed in SEEDS:
        summaries={variant:[source_summary(**{k:pairs[seed][s][variant][k] for k in ('prediction','target','sensitivity')}) for s in order] for variant in ('R0','COISC')}
        for i,source in enumerate(order):
            for variant in ('R0','COISC'):
                a=summaries[variant][i]
                result['prediction']+=_prediction_rows(seed,variant,roles[source],source,'within_source',a['MAE'],a['MSE'],a['bias'],1,a['n'])
                result['raw'].append(dict(seed=seed,source_id=source,role=roles[source],variant=variant,**pairs[seed][source][variant]))
                for o,point in enumerate(POINTS):
                    result['sensitivity'].append(dict(seed=seed,variant=variant,cohort=roles[source],source_id=source,aggregation='within_source',output=point,
                        PLE=a['PLE'][o],WSR=a['WSR'][o],PSM=a['PSM'][o],pre_cells=a['pre_count'][o],post_cells=a['post_count'][o],n_windows=a['n']))
                    for k in range(12):
                        result['sensitivity_distribution'].append(dict(seed=seed,variant=variant,source_id=source,cohort=roles[source],output=point,horizon_s=(k+1)*10,
                            n_windows=a['n'],**{f'S_P{q}':float(np.percentile(a['sensitivity'][:,k,o],q)) for q in (10,25,50,75,90)}))
                    for hi,h in enumerate(HORIZONS):
                        result['sufficient_statistics'].append(dict(seed=seed,variant=variant,source_id=source,output=point,horizon_s=h,n_windows=a['n'],
                            **{k:float(a[k][hi,o]) for k in ('sum_abs_error','sum_squared_error','sum_error')}))
            p0=pairs[seed][source]['R0'];p1=pairs[seed][source]['COISC']
            for hi,h in enumerate(HORIZONS):
                for o,point in enumerate(POINTS):
                    k=h//10-1;e0=finite(p0['prediction'][:,k,o])-finite(p0['target'][:,k,o]);e1=finite(p1['prediction'][:,k,o])-finite(p1['target'][:,k,o])
                    d=np.abs(e1)-np.abs(e0)
                    result['source_audit'].append(dict(seed=seed,source_id=source,cohort=roles[source],horizon_s=h,output=point,n_windows=len(d),
                        R0_MAE_C=float(np.abs(e0).mean()),COISC_MAE_C=float(np.abs(e1).mean()),delta_MAE_C=float(d.mean()),
                        R0_RMSE_C=float(np.sqrt(np.mean(e0**2))),COISC_RMSE_C=float(np.sqrt(np.mean(e1**2))),
                        delta_RMSE_C=float(np.sqrt(np.mean(e1**2))-np.sqrt(np.mean(e0**2))),
                        R0_bias_C=float(e0.mean()),COISC_bias_C=float(e1.mean()),delta_bias_C=float((e1-e0).mean()),
                        improved=int((d<0).sum()),worsened=int((d>0).sum()),tied=int((d==0).sum()),
                        delta_P10=float(np.quantile(d,.1)),delta_P25=float(np.quantile(d,.25)),delta_P50=float(np.quantile(d,.5)),delta_P75=float(np.quantile(d,.75)),delta_P90=float(np.quantile(d,.9))))
        seed_primary=None
        for cohort,ix in cohorts.items():
            counts=draw['counts'][:,ix]
            estimates={}; boots={}
            for variant in ('R0','COISC'):
                ss=[summaries[variant][i] for i in ix]
                n=sum(s['n'] for s in ss)
                arr={k:np.stack([s[k] for s in ss]) for k in ('MAE','MSE','bias')}
                est={k:arr[k].mean(0) for k in arr};boot={k:draw_means(arr[k],counts) for k in arr}
                result['prediction']+=_prediction_rows(seed,variant,cohort,'ALL','equal_source',est['MAE'],est['MSE'],est['bias'],len(ix),n)
                weights=np.array([s['n'] for s in ss]);weights=weights/weights.sum()
                pooled={k:np.tensordot(weights,v,axes=(0,0)) for k,v in arr.items()}
                result['prediction']+=_prediction_rows(seed,variant,cohort,'ALL','window_pooled',pooled['MAE'],pooled['MSE'],pooled['bias'],len(ix),n)
                for metric in ('PLE','WSR','PSM'):
                    est[metric]=[];boot[metric]=[]
                    for o in range(7):
                        if ss[0][metric][o] is None:
                            value=bd=None
                        elif metric=='PSM':
                            calc=SourceMedian([s['post_cells'][o] for s in ss])
                            value=float(calc.many(np.ones((1,len(ix)),dtype=int))[0]);bd=calc.many(counts)
                        else:
                            vals=np.array([s[metric][o] for s in ss]);value=float(vals.mean());bd=draw_means(vals,counts)
                        est[metric].append(value);boot[metric].append(bd)
                for o,point in enumerate(POINTS):
                    pre_cells=sum(s['pre_count'][o] for s in ss);post_cells=sum(s['post_count'][o] for s in ss)
                    result['sensitivity'].append(dict(seed=seed,variant=variant,cohort=cohort,source_id='ALL',aggregation='equal_source',output=point,
                        **{k:est[k][o] for k in ('PLE','WSR','PSM')},pre_cells=pre_cells,post_cells=post_cells,n_windows=n,n_sources=len(ix)))
                    all_s=np.concatenate([s['sensitivity'][:,:,o] for s in ss])
                    pooled_pre=all_s[:,PRE[:,o]];pooled_post=all_s[:,POST[:,o]]
                    result['sensitivity'].append(dict(seed=seed,variant=variant,cohort=cohort,source_id='ALL',aggregation='window_pooled',output=point,
                        PLE=float(np.abs(pooled_pre).mean()) if pre_cells else None,WSR=float((pooled_post < -EPS_SIGN).mean()) if post_cells else None,
                        PSM=float(np.median(np.abs(pooled_post))) if post_cells else None,pre_cells=pre_cells,post_cells=post_cells,n_windows=n,n_sources=len(ix)))
                estimates[variant]=est;boots[variant]=boot
            e0,e1=estimates['R0'],estimates['COISC'];b0,b1=boots['R0'],boots['COISC']
            for hi,h in enumerate(HORIZONS):
                for o,point in enumerate(POINTS):
                    for name,key,sqrt in (('MAE','MAE',False),('RMSE','MSE',True),('bias','bias',False)):
                        f=np.sqrt if sqrt else lambda x:x
                        result['bootstrap'].append(dict(seed=seed,cohort=cohort,output=point,horizon_s=h,endpoint=name,n_sources=len(ix),draw_sha256=draw['sha256'],
                            **effect_row(f(e0[key][hi,o]),f(e1[key][hi,o]),f(b0[key][:,hi,o]),f(b1[key][:,hi,o]))))
            for o,point in enumerate(POINTS):
                for metric in ('PLE','WSR','PSM'):
                    r=dict(seed=seed,cohort=cohort,output=point,horizon_s=None,endpoint=metric,n_sources=len(ix),draw_sha256=draw['sha256'])
                    r.update(effect_row(e0[metric][o],e1[metric][o],b0[metric][o],b1[metric][o]) if e0[metric][o] is not None else dict(status='NA_empty_region',R0=None,COISC=None))
                    result['bootstrap'].append(r)
            # Collateral distribution: report every output, horizon and source.
            a0=np.stack([summaries['R0'][i]['MAE'] for i in ix]);a1=np.stack([summaries['COISC'][i]['MAE'] for i in ix])
            for hi in list(range(5))+[None]:
                for o in list(range(7))+[None]:
                    v0=a0[:,hi:hi+1,:] if hi is not None else a0
                    v1=a1[:,hi:hi+1,:] if hi is not None else a1
                    if o is not None:v0=v0[:,:,o:o+1];v1=v1[:,:,o:o+1]
                    v0=v0.mean((1,2));v1=v1.mean((1,2));delta=v1-v0
                    result['collateral'].append(dict(seed=seed,cohort=cohort,source_id='ALL',output=POINTS[o] if o is not None else 'all7',horizon_s=HORIZONS[hi] if hi is not None else 'all5',
                        n_sources=len(ix),improved=int((delta<0).sum()),worsened=int((delta>0).sum()),tied=int((delta==0).sum()),
                        fraction_improved=float((delta<0).mean()),fraction_worsened=float((delta>0).mean()),fraction_tied=float((delta==0).mean()),
                        min_delta=float(delta.min()),max_delta=float(delta.max()),**{f'delta_P{q}':float(np.percentile(delta,q)) for q in (10,25,50,75,90)},
                        **effect_row(float(v0.mean()),float(v1.mean()),draw_means(v0,counts),draw_means(v1,counts))))
            for j,i in enumerate(ix):
                result['collateral'].append(dict(seed=seed,cohort=cohort,source_id=order[i],output='all7',horizon_s='all5',R0=float(a0[j].mean()),COISC=float(a1[j].mean()),absolute_change=float((a1[j]-a0[j]).mean())))
            if cohort=='all29':
                g={'MAE':dict(R0=float(e0['MAE'][4,6]),COISC=float(e1['MAE'][4,6]),contrast_ci=percentile_ci(b1['MAE'][:,4,6]-b0['MAE'][:,4,6]))}
                for metric in ('PLE','WSR'):
                    g[metric]=dict(R0=e0[metric][6],COISC=e1[metric][6],contrast_ci=percentile_ci(b1[metric][6]-.9*b0[metric][6]))
                g['PSM']=dict(R0=e0['PSM'][6],COISC=e1['PSM'][6],ratio_contrast_ci=percentile_ci(b1['PSM'][6]-.2*b0['PSM'][6]),floor_contrast_ci=percentile_ci(b1['PSM'][6]-.0005))
                seed_primary=dict(seed=seed,draw_sha256=draw['sha256'],primary_values=g,**decide_gates(g))
        # Gate contrasts are exported explicitly, not only hidden in JSON.
        g=seed_primary['primary_values']
        for metric,key,bound,threshold in [('MAE','contrast_ci','upper',.005),('PLE','contrast_ci','upper',0.),('WSR','contrast_ci','upper',0.),('PSM','ratio_contrast_ci','lower',0.),('PSM','floor_contrast_ci','lower',0.)]:
            result['bootstrap'].append(dict(seed=seed,cohort='all29',output='Tp',endpoint=metric,gate_contrast=key,ci_lower=g[metric][key][0],ci_upper=g[metric][key][1],
                CI_bound=bound,threshold=threshold,R0=g[metric]['R0'],COISC=g[metric]['COISC'],n_sources=29,n_replicates=REPLICATES,draw_sha256=draw['sha256']))
        decisions[seed]=seed_primary;result['per_seed'].append(seed_primary)
    status=overall_decision(decisions)
    result['decision']=dict(overall_decision=status,overall_PASS=status=='PASS',per_seed=result['per_seed'],no_rescue=True,
                           collateral_effect_report='frozen_collateral_effect_audit.csv',scope='fixed seeds;source-balanced all29')
    result['status']=status
    return result

def _jsonable(x):
    if isinstance(x,np.ndarray):return _jsonable(x.tolist())
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [_jsonable(v) for v in x]
    return x

def output_payloads(result):
    """Serialize to memory only; no path accepted and no artifacts written."""
    outputs={}
    def j(name,obj):outputs[name]=json.dumps(_jsonable(obj),ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    def c(name,rows):
        fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows([{k:json.dumps(_jsonable(v)) if isinstance(v,(list,dict,np.ndarray)) else v for k,v in r.items()} for r in rows])
        outputs[name]=stream.getvalue().encode('utf-8-sig')
    j('FROZEN_FINAL_DECISION.json',result['decision']);j('per_seed_frozen_metrics.json',result['per_seed'])
    for name,key in [('frozen_prediction_metrics.csv','prediction'),('frozen_sensitivity_metrics.csv','sensitivity'),('frozen_source_horizon_output_audit.csv','source_audit'),('frozen_cluster_bootstrap.csv','bootstrap'),('frozen_collateral_effect_audit.csv','collateral')]:c(name,result[key])
    c('source_sufficient_statistics.csv',result.get('sufficient_statistics',[]))
    c('frozen_sensitivity_distribution.csv',result.get('sensitivity_distribution',[]))
    c('frozen_model_summary.csv',[r for r in result['prediction'] if r['aggregation']=='equal_source' and r['output'] in ('all7','Tp')])
    for name,key in [('frozen_free_run_metrics.csv','free_run'),('frozen_event_details.csv','event_details'),('frozen_event_summary.csv','event_summary'),('frozen_regime_metrics.csv','regime'),('frozen_inference_time.csv','inference_time'),('failure_source_T5_diagnostic.csv','failure_source')]:
        c(name,result['diagnostics'].get(key,[dict(status='not_requested_or_unavailable',primary=False)]))
    raw={};sensitivity={}
    for i,r in enumerate(result['raw']):
        for key in ('window_ids','target','origin','history','future','prediction'):
            raw[f'record_{i}_{key}']=r[key]
        raw[f'record_{i}_identity']=np.array([str(r['seed']),r['source_id'],r['role'],r['variant']])
        error=finite(r['prediction'])-finite(r['target'])
        raw[f'record_{i}_signed_error']=error;raw[f'record_{i}_absolute_error']=np.abs(error);raw[f'record_{i}_squared_error']=error**2
        sensitivity[f'record_{i}_S']=r['sensitivity']
        u=r['future'][:,0,1]
        sensitivity[f'record_{i}_clipped_denominator']=(np.clip(u+25,0,500)-np.clip(u-25,0,500))/500
    sensitivity.update(pre_mask=PRE,post_mask=POST)
    for name,data in [('window_level_raw_results.npz',raw),('window_level_sensitivity.npz',sensitivity)]:
        stream=io.BytesIO();np.savez_compressed(stream,**data);outputs[name]=stream.getvalue()
    if 'draw_table' in result:
        d=result['draw_table'];stream=io.BytesIO();np.savez_compressed(stream,indices=d['indices'],counts=d['counts'],source_order=np.array(d['source_order']),seed=d['seed']);outputs['bootstrap_draws.npz']=stream.getvalue()
    return outputs

if __name__=='__main__':
    run_frozen_evaluation()

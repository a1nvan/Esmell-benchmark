"""Verify selected headline values in the manuscript against frozen outputs."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'results'

def csv(task, name):
    return pd.read_csv(ROOT / task / name)

def verify(label, actual, expected, decimals):
    if round(float(actual), decimals) != round(float(expected), decimals):
        raise AssertionError(f'{label}: got {actual}, expected {expected}')
    print(f'PASS {label}: {float(actual):.{decimals}f}')

s=csv('E1','summary.csv')
for model, card, expected in [('CNN',130,.844),('CNN',146,.587),('CNN',166,.452),('CNN',169,.458),('ExtraTrees',130,.801)]:
    q=s[(s.experiment=='frozen_130')&(s.model==model)&(s.test_card==card)]
    assert len(q)==1
    verify(f'E1 {model} {card} condition-BAcc',q.iloc[0].condition_bacc,expected,3)
q=csv('E1','matched_169_scores.csv')
r=q[q.model=='166 ExtraTrees'].iloc[0]
verify('E1 recalibrated BAcc',r.bacc,.648,3)
assert (r.pentane_correct,r.pentane_total)==(52,97)

q=csv('E2','summary.csv')
for direction, method, expected in [('171->172','source_only',.441),('172->171','source_only',.386),('171->172','target_reference',.806),('172->171','target_reference',.817)]:
    v=q[(q.direction==direction)&(q.method==method)]
    assert len(v)==1
    verify(f'E2 {direction} {method}',v.iloc[0].bacc_mean,expected,3)

q=csv('E3','table_E3.csv')
v=q[q.variant=='fixed_hh'].iloc[0]
verify('E3 reference device 171 BAcc',v['171_test'],.639,3)
verify('E3 reference device 172 BAcc',v['172_test'],.791,3)

q=csv('E4','scores.csv')
for card,model,expected in [(171,'ROCKET',.690),(172,'LogReg64',.824)]:
    v=q[(q.card==card)&(q.model==model)&(q.protocol=='future')]
    assert len(v)==1
    verify(f'E4 {card} {model} future BAcc',v.iloc[0].bacc,expected,3)

q=csv('E5','E5_diagnostic_summary.csv')
for model, metric, expected in [('ExtraTrees8','macro8',.252),('ExtraTrees8','mix_exact',0),('ROCKET','mix_exact',.259)]:
    v=q[(q.model==model)&(q.split=='test')]
    assert len(v)==1
    verify(f'E5 {model} {metric}',v.iloc[0][metric],expected,3)

q=csv('E6','scores.csv')
for model,expected in [('GasMedian',.302),('ExtraTrees',.358)]:
    v=q[(q.model==model)&(q.split=='test')]
    assert len(v)==1
    verify(f'E6 {model} Macro-6',v.iloc[0].macro6_log_mae,expected,3)
v=q[(q.model=='ConvAttention')&(q.split=='test')]
assert len(v)==3
verify('E6 ConvAttention Macro-6 seed mean',v.macro6_log_mae.mean(),.337,3)

q=csv('E6', 'retrospective_protocols.csv').set_index('protocol')
for key, median, trees in [
    ('day_grouped_cv', .291, .224),
    ('next_day_130', .181, .324),
    ('later_campaign_146', .302, .376),
]:
    verify(f'E6 retrospective {key} median', q.loc[key, 'GasMedian'], median, 3)
    verify(f'E6 retrospective {key} ExtraTrees', q.loc[key, 'ExtraTrees'], trees, 3)

q=csv('E7','scores.csv')
v=q[(q.model=='ExtraTrees3+Median')&(q.split=='future146')]
assert len(v)==1
verify('E7 ExtraTrees3+Median normalized MAE',v.iloc[0].e2e_macro8_nmae,.230,3)
v=q[(q.model=='CNN_joint+Median')&(q.split=='future146')]
assert len(v)==3
verify('E7 joint CNN mixture exact-match mean',v.exact_mix.mean(),.126,3)
print('PASS: selected headline results agree with frozen tables.')

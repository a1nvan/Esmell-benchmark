from pathlib import Path
import hashlib, json, re, sys
ROOT = Path(__file__).resolve().parents[1]
def checksum(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

manifest=ROOT/'MANIFEST.sha256'
count=0
for line in manifest.read_text(encoding='utf-8').splitlines():
    expected,name=line.split('  ',1)
    path=ROOT/name
    assert path.is_file(),f'Missing: {name}'
    assert checksum(path)==expected,f'Changed output or preprocessing source: {name}'
    count+=1
nbs=sorted((ROOT/'experiments').glob('*.ipynb'))
assert len(nbs)==7,f'Expected 7 notebooks, found {len(nbs)}'
for p in nbs:
    d=json.loads(p.read_text(encoding='utf-8'))
    assert d['nbformat']==4
    for i,c in enumerate(d['cells']):
        if c['cell_type']=='code':
            assert c.get('execution_count') is None and not c.get('outputs')
            compile(''.join(c['source']),f'{p.name}:{i}','exec')
            text=''.join(c['source'])
            assert not any(x in text for x in ('C:\\\\', '/Users/', '/home/')),f'Local path in {p.name}:{i}'
for name in ['E1','E2','E3','E4','E5','E6','E7']:
    assert any((ROOT/'results'/name).rglob('*.csv')),f'Missing results: {name}'
assert not any((ROOT/'results').rglob('*.pt'))
assert not any((ROOT/'results').rglob('*.joblib'))
assert not any((ROOT/'results').rglob('*.npz'))
extra=ROOT/'scripts'/'reproduce_e6_retrospective.py'
compile(extra.read_text(encoding='utf-8'),str(extra),'exec')
for artifact in ('retrospective_protocols.csv','retrospective_by_gas_and_type.csv',
                 'retrospective_predictions.csv','retrospective_run_config.json'):
    assert (ROOT/'results'/'E6'/artifact).is_file(),f'Missing E6 control: {artifact}'
print(f'PASS: {len(nbs)} notebooks compiled; {count} frozen result/source checksums verified.')

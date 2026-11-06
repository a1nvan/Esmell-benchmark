"""Restore the three unchanged full prediction CSVs from the release bundle."""
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'compressed_results'/'full_prediction_tables.zip'
ALLOWED={
 'results/E5/predictions_all_splits.csv',
 'results/E6/predictions_long.csv',
 'results/E7/predictions.csv',
}
with ZipFile(ARCHIVE) as z:
    assert set(z.namelist())==ALLOWED, 'Unexpected bundled files'
    for name in sorted(ALLOWED):
        output=ROOT/name
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_bytes(z.read(name))
        print('Restored:',name)
print('Now run: python scripts/verify_artifact.py')

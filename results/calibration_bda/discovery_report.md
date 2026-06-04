# Auto-discovery report
generated: 2026-05-28T14:44:28.819116+00:00
scanned_valid_csv: 2
candidates: 2

- [OK] data\raw\03_hudson_bay_lynx_hare\lynxhare.csv
- [OK] data\raw\07_lter_fish\WIfishAbundance.csv

## Detected predator-prey series
- **lynxhare.csv** (lynx_hare) conf=0.95 method=lynx_hare_signature
  - time=`year` prey=`hare` pred=`lynx` group=-
  - classic hudson bay or lynx-roedeer style
- **WIfishAbundance.csv** (lter_fish) conf=0.88 method=lter_fish_pair
  - time=`YEAR` prey=`Bluegill` pred=`Largemouth Bass` group=WBIC=804600
  - Bluegill vs Largemouth Bass, lake=804600, n_years=9
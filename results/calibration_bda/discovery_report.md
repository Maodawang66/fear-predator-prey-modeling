# Auto-discovery report
generated: 2026-06-11T14:03:01.138742+00:00
scanned_valid_csv: 4
candidates: 12

- [OK] data\raw\01_lynx_roe_deer\Andren_lynx_roedeer_data.csv
- [OK] data\raw\02_killifish_mosquitofish\TimeSeriesLogMeans.csv
- [OK] data\raw\04_lake_michigan_zooplankton\GLERL_M110_Zoop_1994-2012.txt
- [OK] data\raw\07_lter_fish\WIfishAbundance.csv

## Detected predator-prey series
- **Andren_lynx_roedeer_data.csv** (lynx_roe) conf=0.92 method=lynx_roe_by_region
  - time=`year` prey=`roe_deer_harvest_mean` pred=`lynx_family_groups` group=1
  - region=1, density via area
- **Andren_lynx_roedeer_data.csv** (lynx_roe) conf=0.92 method=lynx_roe_by_region
  - time=`year` prey=`roe_deer_harvest_mean` pred=`lynx_family_groups` group=2
  - region=2, density via area
- **Andren_lynx_roedeer_data.csv** (lynx_roe) conf=0.92 method=lynx_roe_by_region
  - time=`year` prey=`roe_deer_harvest_mean` pred=`lynx_family_groups` group=3
  - region=3, density via area
- **Andren_lynx_roedeer_data.csv** (lynx_roe) conf=0.92 method=lynx_roe_by_region
  - time=`year` prey=`roe_deer_harvest_mean` pred=`lynx_family_groups` group=4
  - region=4, density via area
- **Andren_lynx_roedeer_data.csv** (lynx_roe) conf=0.92 method=lynx_roe_by_region
  - time=`year` prey=`roe_deer_harvest_mean` pred=`lynx_family_groups` group=5
  - region=5, density via area
- **Andren_lynx_roedeer_data.csv** (lynx_roe) conf=0.92 method=lynx_roe_by_region
  - time=`year` prey=`roe_deer_harvest_mean` pred=`lynx_family_groups` group=6
  - region=6, density via area
- **Andren_lynx_roedeer_data.csv** (lynx_roe) conf=0.92 method=lynx_roe_by_region
  - time=`year` prey=`roe_deer_harvest_mean` pred=`lynx_family_groups` group=7
  - region=7, density via area
- **TimeSeriesLogMeans.csv** (killifish) conf=0.90 method=killifish_by_site
  - time=`DATESEQ` prey=`ME2LOGHETADS` pred=`ME4LOGGAMBO` group=TP
  - site=TP, log->linear
- **TimeSeriesLogMeans.csv** (killifish) conf=0.90 method=killifish_by_site
  - time=`DATESEQ` prey=`ME2LOGHETADS` pred=`ME4LOGGAMBO` group=WRGP
  - site=WRGP, log->linear
- **TimeSeriesLogMeans.csv** (killifish) conf=0.90 method=killifish_by_site
  - time=`DATESEQ` prey=`ME2LOGHETADS` pred=`ME4LOGGAMBO` group=WRHW
  - site=WRHW, log->linear
- **GLERL_M110_Zoop_1994-2012.txt** (zooplankton) conf=0.93 method=zooplankton_glerl
  - time=`Date` prey=`D.mendotae` pred=`Bythotrephes` group=-
  - Marino et al. Lake Michigan CE/NCE monitoring
- **WIfishAbundance.csv** (lter_fish) conf=0.88 method=lter_fish_pair
  - time=`YEAR` prey=`Bluegill` pred=`Largemouth Bass` group=WBIC=804600
  - Bluegill vs Largemouth Bass, lake=804600, n_years=9
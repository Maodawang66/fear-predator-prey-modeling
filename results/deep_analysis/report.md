# Deep data analysis report
generated: 2026-06-01T06:07:02.196163+00:00

## Tier 1
- cross_system rows: 12
- RMSE improvement entries: 12
- k identifiability entries: 12

### Top RMSE improvements
- timeserieslogmeans_WRHW: baseline/bda = 3.4e+06
- lynxhare: baseline/bda = 6.3e+05
- timeserieslogmeans_WRGP: baseline/bda = 3.5e+04
- andren_lynx_roedeer_data_5: baseline/bda = 3.5e+03
- timeserieslogmeans_TP: baseline/bda = 2.2e+03

## Tier 2
- Peacor: {'n_studies': 64, 'effect_col': 'Predation effect', 'effect_type': 'categorical', 'TMIE_count': 37, 'NCE_count': 27, 'tmie_fraction': 0.578125, 'by_taxon': {'Invertebrate': {'TMIE': 11, 'NCE': 1}, 'Vertebrate': {'TMIE': 26, 'NCE': 26}}}
- Coral reef max foraging suppression: 0.0
- Damselfly cage suppression: -0.06328181607773908
- LTER extra fits: 0

## Figures
- tier1/rmse_improvement.png
- tier1/eta_by_group.png
- tier1/killifish_sites.png
- tier1/andren_regions.png
- tier1/k_profile_*.png
- tier2/peacor_effect_distribution.png
- tier2/peacor_by_taxon.png
- tier2/coral_foraging_by_position.png
- tier2/damselfly_activity_by_treatment.png
- tier2/mechanism_prior_comparison.png
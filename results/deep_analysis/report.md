# Deep data analysis report
generated: 2026-06-05T16:52:19.112254+00:00

## Tier 1
- cross_system rows: 12
- RMSE improvement entries: 12
- k identifiability entries: 3

### Top RMSE improvements
- andren_lynx_roedeer_data_2: holdout baseline/bda = 6.7; best validation=bda_fear; best AICc=bda_fear
- andren_lynx_roedeer_data_4: holdout baseline/bda = 5; best validation=bda_fear; best AICc=bda_fear
- andren_lynx_roedeer_data_6: holdout baseline/bda = 2.8; best validation=fear_memory; best AICc=bda_fear
- andren_lynx_roedeer_data_1: holdout baseline/bda = 2.2; best validation=bda_fear; best AICc=bda_fear
- andren_lynx_roedeer_data_3: holdout baseline/bda = 1.7; best validation=bda_fear; best AICc=bda_fear

## Tier 2
- Peacor: {'n_studies': 64, 'effect_col': 'Predation effect', 'effect_type': 'categorical', 'TMIE_count': 37, 'NCE_count': 27, 'tmie_fraction': 0.578125, 'by_taxon': {'Invertebrate': {'TMIE': 11, 'NCE': 1}, 'Vertebrate': {'TMIE': 26, 'NCE': 26}}}
- Coral reef herbivory suppression: 0.5619047618999999
- Damselfly cage suppression: -0.06328181607773908
- LTER extra fits: 3

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
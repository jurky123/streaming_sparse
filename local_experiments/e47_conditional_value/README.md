# E47 frozen P_set and actual conditional value

`python -m local_experiments.e47_conditional_value.run --seed 17` (and29), then `.report`.
No training. P is E44 P_set; L is old E43 L reference. Same locked E45 queries, masks and candidate pools. True action matrices match E45 exactly. P_set original96 scores replay E44 exactly.

V_adaptive versus V_fixedplus uses equal event-state weights; V_fixedplus allows perfect acceptance of one fixed action and is an offline diagnostic, not a router. Complete matrices and frozen replay source/models are provided in review artifacts; Test sealed.

RESULT_DICT = {
    'inat': {
        'default': { # The original split
            'supervised': {
                'default': { # The lifting mode name
                    # Val set: Head: 58.930%, Tail: 46.377%, Mean: 52.561%
                    # Test set: Head: 64.435%, Tail: 50.559%, Mean: 57.394%
                    # pl: Average of tail accs: 0.8856448531150818
                    # pl_entropy: Average of tail accs: 0.8057585954666138
                    "wd_1e_4": {'result': "checkpoints/supervised/inat/09-14-2022-02:24"},
                    # Val set: Head: 57.933%, Tail: 42.751%, Mean: 50.230%
                    # Test set: Head: 63.285 %, Tail: 48.048%, Mean: 55.554 %
                    "wd_1e_3": {'result': "checkpoints/supervised/inat_wd_1e_3/11-23-2022-17:26"},
                    # Val set: Head: 58.854%, Tail: 46.047%, Mean: 52.356%
                    # Test set: Head: 64.690%, Tail: 51.108%, Mean: 57.798%
                    # pl: Average of tail accs: 0.8828063607215881
                    # pl_entropy: Average of tail accs: 0.8065689206123352
                    "wd_1e_5": {'result': "checkpoints/supervised/inat_wd_1e_5/11-23-2022-17:26"},
                },
                "default_lift_uniform": {
                    "wd_1e_4": {'result': "checkpoints/supervised/inat_lift_uniform/10-03-2022-12:29"}, # Best Val Acc 0.548197 | Best Test Acc 0.6018
                    "wd_1e_3": {'result': "checkpoints/supervised/inat_lift_uniform_wd_1e_3/11-24-2022-01:39"}, # Best Val Acc 0.53482 | Best Test Acc 0.58921
                    "wd_1e_5": {'result': "checkpoints/supervised/inat_lift_uniform_wd_1e_5/11-24-2022-01:52"}, # Best Val Acc 0.55490 | Best Test Acc 0.60818
                },
                "default_lift_tail_uniform": {
                    "wd_1e_4": {'result': "checkpoints/supervised/inat_lift_tail_uniform/10-01-2022-22:09"}, # Best Val Acc 0.570426 | Best Test Acc 0.62982
                    "wd_1e_3": {'result': "checkpoints/supervised/inat_lift_tail_uniform_1e_3/11-24-2022-12:18"}, # Best Val Acc 0.551686 | Best Test Acc 0.61014
                    "wd_1e_5": {'result': "checkpoints/supervised/inat_lift_tail_uniform_1e_5/11-24-2022-12:19"}, # Best Val Acc 0.56968 | Best Test Acc 0.62779986858
                },
            },
            'fixmatch': {
                'default': {
                    # Val set: Head: 68.487%, Tail: 61.803%, Mean: 65.095%
                    # Test set: Head: 74.827 %, Tail: 66.431%, Mean: 70.567 %
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/inat_fixmatch/09-16-2022-11:30"},
                },
                'default_lift_uniform': {
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/inat_fixmatch_lift_uniform/10-04-2022-15:20"}, # Best Val Acc 0.65856 | Best Test Acc 0.7253657579
                },
                'default_lift_tail_uniform': {
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/inat_fixmatch_lift_tail_uniform/10-03-2022-12:24"}, # Best Val Acc 0.666725 | Best Test Acc 0.73369
                }
            },
            'debiased': {
                'default': {
                    # Val set: Head: 69.066%, Tail: 60.527%, Mean: 64.733%
                    # Test set: Head: 75.719 %, Tail: 66.452%, Mean: 71.017 %
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/inat_debiased_tau_4/10-09-2022-13:24"},
                },
                'default_lift_uniform': {
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/inat_debiased_tau_4_lift_uniform/10-12-2022-14:38"}, # Best Val Acc 0.6691 | Best Test Acc 0.7259998
                },
                'default_lift_tail_uniform': {
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/inat_debiased_tau_4_lift_tail_uniform/10-12-2022-14:35"}, # Best Val Acc 0.664375 | Best Test Acc 0.73205
                }
            },
        },
    },
    'imagenet127': {
        'default': {
            'supervised': {
                'default': {
                    # Val set: Head: 52.286%, Tail: 44.621%, Mean: 48.303%
                    # Test set: Head: 48.147 %, Tail: 35.636%, Mean: 41.645 %
                    # PL retrieval: Average of tail accs: 0.8648990392684937
                    # PL+entropy retrieval: Average of tail accs: 0.8712121844291687
                    "wd_1e_4": {'result': "checkpoints/supervised/imagenet127/09-16-2022-11:48"},
                    "wd_1e_3": {'result': "checkpoints/supervised/imagenet127_wd_1e_3/11-23-2022-22:10"}, # Best Val Acc 0.3994 | Best Test Acc 0.3539855
                    "wd_1e_5": {'result': "checkpoints/supervised/imagenet127_wd_1e_5/11-24-2022-17:36"}, # Best Val Acc 0.5044 | Best Test Acc 0.43532
                },
                "default_lift_uniform": {
                    # Best Val Acc 0.4110061824321747 @ epoch 261
                    # Best Test Acc 0.3684348464012146 @ Best val epoch 261
                    "wd_1e_4": {'result': "checkpoints/supervised/imagenet127_lift_uniform/11-25-2022-23:19"}, # Best Val Acc 0.492768 | Best Test Acc 0.43945
                    "wd_1e_3": {'result': "checkpoints/supervised/imagenet127_lift_uniform_wd_1e_3/11-29-2022-01:38"}, # Best Val Acc 0.4110061824321747 | Best Test Acc 0.3684348464012146
                    "wd_1e_5": {'result': "checkpoints/supervised/imagenet127_lift_uniform_wd_1e_5/11-25-2022-23:22"}, # Best Val Acc 0.499795 | Best Test Acc 0.44753
                },
                "default_lift_tail_uniform": {
                    "wd_1e_4": {'result': "checkpoints/supervised/imagenet127_lift_tail_uniform/10-03-2022-12:16"}, # Best Val Acc 0.55139 | Best Test Acc 0.49057
                    "wd_1e_3": {'result': "checkpoints/supervised/imagenet127_lift_tail_uniform_1e_3/11-25-2022-01:33"}, # Best Val Acc 0.50956 | Best Test Acc 0.453277
                    "wd_1e_5": {'result': "checkpoints/supervised/imagenet127_lift_tail_uniform_1e_5/11-24-2022-22:37"}, # Best Val Acc 0.55847 | Best Test Acc 0.4878
                },
            },
            'fixmatch': {
                'default': {
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_fixmatch_1/09-25-2022-11:27"}, # Best Val Acc 0.62972 | Best Test Acc 0.5753859
                },
                'default_lift_uniform': {
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_fixmatch_1_lift_uniform/11-27-2022-17:42"}, # Best Val Acc 0.63286 | Best Test Acc 0.581787
                },
                'default_lift_tail_uniform': {
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_fixmatch_1_lift_tail_uniform/10-05-2022-14:55"}, # Best Val Acc 0.65858 | Best Test Acc 0.61079
                }
            },
            'debiased': {
                'default': {
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_debiased_1/09-21-2022-01:11"}, # Best Val Acc 0.7203 | Best Test Acc 0.6712577
                },
                'default_lift_uniform': {
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_debiased_1_lift_uniform/10-05-2022-14:55"}, # Best Val Acc 0.731797 | Best Test Acc 0.67049
                },
                'default_lift_tail_uniform': {
                    "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_debiased_1_lift_tail_uniform/10-04-2022-15:085"}, # Best Val Acc 0.74146658 | Best Test Acc 0.6936
                },
            },
        },
    },
}

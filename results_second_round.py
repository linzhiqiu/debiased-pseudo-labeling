RESULT_DICT = {
    'inat': {
        'default': { # The original split
            'supervised': {
                'default': { # The lifting mode name
                    "wd_1e_5": {
                        "default_lift_supervised_wd_1e_5_pl_entropy_all_thre_18_budget_6": "checkpoints/supervised/inat_default_lift_supervised_wd_1e_5_pl_entropy_all_thre_18_budget_6_lr_00003_wd_1e_5/01-11-2023-18:20",
                        # 'default_lift_supervised_wd_1e_5_pl_entropy_all_thre_18_budget_6': "checkpoints/supervised/inat_default_lift_supervised_wd_1e_5_pl_entropy_all_thre_18_budget_6_lr_00003_wd_1e_5/12-30-2022-17:18/",
                        'default_lift_supervised_wd_1e_5_pl_entropy_all_tp_thre_18_budget_6': "checkpoints/supervised/inat_default_lift_supervised_wd_1e_5_pl_entropy_all_tp_thre_18_budget_6_lr_00003_wd_1e_5/01-08-2023-23:58/",
                        "default_lift_supervised_wd_1e_5_pl_entropy_all_fp_thre_18_budget_6": "checkpoints/supervised/inat_default_lift_supervised_wd_1e_5_pl_entropy_all_fp_thre_18_budget_6_lr_00003_wd_1e_5/01-09-2023-22:10",
                        'default_lift_supervised_wd_1e_5_pl_thre_18_budget_6': "checkpoints/supervised/inat_default_lift_supervised_wd_1e_5_pl_thre_18_budget_6_lr_00003_wd_1e_5/12-10-2022-01:04/",
                        'default_lift_supervised_wd_1e_5_pl_entropy_thre_18_budget_6': "checkpoints/supervised/inat_default_lift_supervised_wd_1e_5_pl_entropy_thre_18_budget_6_lr_00003_wd_1e_5/12-10-2022-01:03",
                        'default': "checkpoints/supervised/inat_default_from_supervised_lr_00003_wd_1e_5/12-10-2022-23:15",
                    },
                },
            },
            'fixmatch': {
                'default': {
                    "wd_1e_4": {
                        'default_lift_fixmatch_wd_1e_4_pl_thre_18_budget_6': "checkpoints/supervised/inat_default_lift_fixmatch_wd_1e_4_pl_thre_18_budget_6_lr_00003_wd_1e_5/12-10-2022-23:22",
                        'default_lift_fixmatch_wd_1e_4_pl_entropy_thre_18_budget_6': "checkpoints/supervised/inat_default_lift_fixmatch_wd_1e_4_pl_entropy_thre_18_budget_6_lr_00003_wd_1e_5/12-10-2022-23:17",
                        'default': "checkpoints/supervised/inat_default_from_fixmatch_lr_0003_wd_1e_5/11-29-2022-13:02",
                    },
                },
            },
            'debiased': {
                'default': {
                    "wd_1e_4": {
                        'default_lift_debiased_wd_1e_4_pl_thre_18_budget_6': "checkpoints/supervised/inat_default_lift_debiased_wd_1e_4_pl_thre_18_budget_6_lr_00003_wd_1e_5/12-02-2022-00:57",
                        'default_lift_debiased_wd_1e_4_pl_entropy_thre_18_budget_6': "checkpoints/supervised/inat_default_lift_debiased_wd_1e_4_pl_entropy_thre_18_budget_6_lr_00003_wd_1e_5/12-10-2022-23:02",
                        'default': "checkpoints/supervised/inat_default_from_debiased_lr_00003_wd_1e_5/11-29-2022-01:37",
                    },
                },
            },
        },
    },
    'imagenet127': {
        'default': {
            'supervised': {
                'default': {
                    "wd_1e_4": {
                        'default_lift_supervised_wd_1e_5_pl_thre_25_budget_12': "checkpoints/supervised/imagenet127_wd_1e_5_default_lift_supervised_wd_1e_5_pl_thre_25_budget_12_lr_00003_wd_1e_5/11-28-2022-01:11",
                        'default_lift_supervised_wd_1e_5_pl_entropy_thre_25_budget_12': "checkpoints/supervised/imagenet127_wd_1e_5_default_lift_supervised_wd_1e_5_pl_entropy_thre_25_budget_12_lr_00003_wd_1e_5/11-29-2022-01:39",
                    },
                },
            },
            # 'fixmatch': {
            #     'default': {
            #         "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_fixmatch_1/09-25-2022-11:27"}, # Best Val Acc 0.62972 | Best Test Acc 0.5753859
            #     },
            #     'default_lift_uniform': {
            #         "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_fixmatch_1_lift_uniform/11-27-2022-17:42"}, # Best Val Acc 0.63286 | Best Test Acc 0.581787
            #     },
            #     'default_lift_tail_uniform': {
            #         "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_fixmatch_1_lift_tail_uniform/10-05-2022-14:55"}, # Best Val Acc 0.65858 | Best Test Acc 0.61079
            #     }
            # },
            # 'debiased': {
            #     'default': {
            #         "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_debiased_1/09-21-2022-01:11"}, # Best Val Acc 0.7203 | Best Test Acc 0.6712577
            #     },
            #     'default_lift_uniform': {
            #         "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_debiased_1_lift_uniform/10-05-2022-14:55"}, # Best Val Acc 0.731797 | Best Test Acc 0.67049
            #     },
            #     'default_lift_tail_uniform': {
            #         "wd_1e_4": {'result': "checkpoints/semi_supervised/imagenet127_debiased_1_lift_tail_uniform/10-04-2022-15:085"}, # Best Val Acc 0.74146658 | Best Test Acc 0.6936
            #     },
            # },
        },
    },
}

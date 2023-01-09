import argparse
from utils.config import process_config

from agents.sleep_test import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    config_list = [
    # "./configs/shhs/reconstruction/vae_eeg.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json"

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_summation.json"

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t025.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al0_shared_b16_t025.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_t025.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_t033.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t025.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t045.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_onlyi_t25.json",

    # "./configs/shhs/single_channel/fourier_transformer_cls_emg_mat.json"
        
    # "./configs/shhs/multi_modal/eeg_eog_emg/established_models/fourier_transformer_eeg_eog_emg_mat_BIOBLIP_twomode_caouter_al1_shared.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al0_shared.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_vae_BIOBLIP.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_vae.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_sepouter.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al025_randommode.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eog_mat.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_dimproj1k.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP.json",,
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_COCA_3fc.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_adv.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_outerplus_rpos_adv.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_outer.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_bottleneck_lim0.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_late_glearnedbiasedm_outerplus.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_merged.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_οbottleneck.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_οbottleneck_adv_rpos_v2.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_sceloss.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_singlemulti.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_simple.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_outer_rpos_simple.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_shared.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_limited_2.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_limited_2_b32.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_outer.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_2fc.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_order.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_COCA_sep_multisupervised.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_sep_multisupervised.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_sep_combined_multisupervised.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw_frozen_LE.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw_frozen.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw_pretrainedNCH_onlyalign_LE.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw_masked.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_double.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_simple.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_rpos_adv_shared.json"
    # "./configs/shhs/myprepro/single_channel/fourier_transformer_eeg_mat_rpos_adv.json"

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t01.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_notrain.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_onlyal.json",

# "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al1_shared_b16_freetrain.json"

    # "./configs/shhs/single_channel/time_cnn_eeg_tom.json",
    "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_nopos_tom.json",
    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_lossw_pretrainedNCH_onlyalign.json"
    # "./configs/shhs/router/router_fourier_tf_eeg_eog_endtoend.json",
    # "./configs/shhs/single_channel/time_cnn_transformer_eeg.json"
    # "./configs/shhs/single_channel/time_cnn_transformer_eeg_lnfirst.json"
    # "./configs/shhs/single_channel/fourier_transformer_cls_emg_mat.json",
    # "./configs/shhs/router/router_fourier_tf_eeg_eog.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eog_mat.json"
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_overparam.json"
    ]
    finals = []
    for i in config_list:
        config = process_config(i)
        agent_class = globals()[config.agent]
        agent = agent_class(config)
        # agent.check_energies_per_class()
        agent.run()
        agent.finalize()
        del agent

main()
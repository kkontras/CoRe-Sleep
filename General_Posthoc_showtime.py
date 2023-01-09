import sys
import os
import numpy as np

sys.exc_info()
os.chdir('/users/sista/kkontras/Documents/Sleep_Project/')


from posthoc.Helpers.Helper_Importer import Importer
from posthoc.Helpers.Helper_Validator import Validator
from posthoc.Helpers.Helper_LogsPlotter import LogsPlotter
from posthoc.Helpers.Helper_Generate_n_Compare import Generate_n_Compare

config_list = [
    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_merged.jδson",
    # "./configs/shhs/reconstruction/vae_eeg.json",
    # "./configs/shhs/reconstruction/vae_eοg.json",

    # "./configs/shhs/single_channel/fourier_transformer_cls_eeg_mat_adv.json",
    # "./configs/shhs/single_channel/fourier_transformer_eeg_long.json",
    # "./configs/shhs/single_channel/fourier_transformer_cls_eog_mat.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_merged_rpos_adv.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_late_rpos_adv_temp.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al0_shared.json",

    # "./configs/nch/multi_modal/established_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_limited.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_VAE.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_fullca.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t025.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_onlyi_t25.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_t033.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_notrain.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b32_mult_notrain.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_multilr.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3_w.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3_clean.json",

    # "./configs/shhs/multi_modal/eeg_eog_emg/established_models/fourier_transformer_eeg_eog_emg_mat_BIOBLIP_twomode_caouter_al1_shared.json"

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_onlyi.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_early_concat_seponlyi.json",

    # "configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_onlyal.json",
    # "configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_onlyal_same.json",

    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al01_shared_b16_mult_freetrain_lr3_cliplike.json",
    # "./configs/shhs/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BIOBLIP_twomode_caouter_al1_shared_b16_mult_freetrain_lr3_cliplike_prevv.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al1_shared_b16_freetrain.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al0_shared_b16_freetrain.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al1_shared_b16_freetrain_nopos.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_nonshared_b16_freetrain.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_bigal01_shared_b16_freetrain.json",
    #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_noscheduler.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_only_bigal01_shared.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_al01_mod_nopos.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_noscheduler.json"

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop0.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop005.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop01-02.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop02.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop04.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_noal.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_noal_trainmore.json"

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_nomult.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_nomult_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_nomult_al01.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_nomult_al01_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_valal.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_trial3.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_nomult_al01_rposal.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_nomult_al01_aligninner.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_rposal_trial2.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_nonshared_rposal_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_rposal_aligninner_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_aligninner_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_nonshared_b16_freetrain_nopos_drop03_aligninner_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial3.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_augmentwithin_drop005.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_augmentwithin.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_augmentwithin_drop02.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_augmentwithin_drop03.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_augmentrepeat.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpred.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_sharednonbatch.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b32_freetrain_nopos_trial3.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_smallerclassifier128.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_smallerclassifier256.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpred_drop03_classfier256.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpred_drop03.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_Late_al01_mod_nopos.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_Late_al01_mod_nopos_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_Late_singleloss_nopos.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_Late_singleloss_al01_nopos.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_Late_singleloss_al01_nopos_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_Late_multloss_noal_nopos.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_Late_multloss_noal_nopos_trial2.json",
    # #
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_mult.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_nomult.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_nomult_al01.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_al01_mod_nopos.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_mult_t2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_nomult_t2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_nomult_al01_t2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_al01_mod_nopos_t2.json",

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_3pass_al01_mod_nopos.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_nopos.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_trained.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_sinusoidal.json"

    # "./configs/shhs/xsleepnet/fourier_time_xsleepnet_eeg.json",

    # "./configs/sleep_edf/multi_modal/eeg_eog/established_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json",
    # "./configs/sleep_edf/single_channel/fourier_transformer_eeg_mat.json",
    # "./configs/sleep_edf/single_channel/fourier_transformer_eeg_mat_drop01.json",
    # "./configs/sleep_edf/single_channel/fourier_transformer_eeg_mat_drop03.json",
    # "./configs/sleep_edf/single_channel/fourier_transformer_eeg_mat_drop03_5folds.json"
    # "./configs/sleep_edf/single_channel/fourier_transformer_eeg_mat_drop01_smallernet.json",

    # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json",
    # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nowei.json",
    # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout03_nowei.json"
    # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout01_nowei_possin.json",
    # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout01_nowei_possin_trial2.json",
    # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout01_nowei_possin_trial3.json",
    # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout01_nowei_possin_trial4.json",
    # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout01_nowei_possin_trial5.json",
    # "./configs/sleep_edf/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_dropout01_nowei_possin_pretrainedal.json"

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_nomult_al01_rposal.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_rposal_trial2.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_nonshared_b16_freetrain_nopos_drop03_aligninner_trial2.json"
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial3.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_drop03_nomult_al01_aligninner_trial1.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_nomult_al01_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_nomult_al01_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_nomult_al01_aligninner_trial2.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_nomult_al01_aligninner_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_nomult_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_nomult_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_nomult_aligninner_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_mult_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_mult_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_mult_aligninner_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al5_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al5_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al10_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al10_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al40_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al40_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al60_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al60_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al80_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al80_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al100_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_al100_trial1_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_sharedall_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_sharedall_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_sharedall_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_v2_trial2.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_v2_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_drop03_aligninner_v2_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_trial2_pt.json",
    # #
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_drop03_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_drop03_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_drop03_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eog_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eog_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eog_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_aligninner_trial2.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_aligninner_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_nomult_al01_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_nomult_al01_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_nomult_al01_aligninner_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_nomult_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_nomult_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_nomult_aligninner_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_mult_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_mult_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Late_mult_aligninner_trial2_pt.json",

    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_trial1_big.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_al01_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_al01_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_al01_aligninner_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_aligninner_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_mult_aligninner_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_mult_aligninner_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_mult_aligninner_trial2_pt.json",

    "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_v2_trial1.json",
    "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_v2_trial1_pt.json",
    "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_aligninner_v2_trial2_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_al01_aligninner_v2_trial1.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_al01_aligninner_v2_trial1_pt.json",
    # "./configs/paper_finals/shhs/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos_nonsharedpreds_Early_nomult_al01_aligninner_v2_trial2_pt.json",

    # "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_nonsharedpreds_drop01_aligninner_possin_sleepyco.json",
    # "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_nonsharedpreds_drop01_aligninner_possin_trial5.json",
    # "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_nonsharedpreds_drop01_aligninner_possin_trial4.json",
    # "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_nonsharedpreds_drop01_aligninner_nopos_pretrained_trial5.json"

    # "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_nonsharedpreds_drop01_aligninner_possin_trial2.json",
    # "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_sharedpreds_drop03_aligninner_possin_trial2.json",
    # "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_sharedpreds_drop03_aligninner_nopos_trial2.json",
    # "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_nonsharedpreds_drop03_aligninner_possin_trial2.json",
    # "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_sharedpreds_drop03_aligninner_possin_small_trial2.json",
    # "./configs/paper_finals/sleep-edf78/fourier_transformer_eeg_eog_mat_unimodal_eeg_b16_drop03_nopos_trial2.json"

    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al1_pretrain.json",
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al1_pretrain_big.json",

    # "./configs/nch/multi_modal/paper_models/fourier_transformer_eeg_eog_mat_BLIP_al01_shared_b16_freetrain_nopos.json"
    # "./configs/nch/multi_modal/paper_models/fourier_transformer_eeg_eog_mat_EarlyConcat_3pass_al01_mod_nopos.json",
    # "./configs/nch/multi_modal/paper_models/fourier_transformer_eeg_eog_mat_Late_al01_mod_nopos.json",
    # "./configs/nch/multi_modal/paper_models/fourier_transformer_eeg_nopos.json",
    # "./configs/nch/multi_modal/paper_models/fourier_transformer_eog_nopos.json",

    # "./configs/shhs/single_channel/time_cnn_eeg_tom.json"
    # "./configs/shhs/multi_modal/eeg_eog/paper_models/fourier_transformer_eeg_nopos_tom.json",

]
count = 0
multi_fold_results = {}

eeg, eog, mm = [], [], []
eeg_std, eog_std, mm_std = [], [], []
for i, config_name in enumerate(config_list):
    for fold in range(1):
        # fold = len(config_list)*i + fold
        importer = Importer(config_name=config_name, device="cuda:0", fold=fold)
        try:
            importer.load_checkpoint()
        except:
            print("We could not load {}".format(config_name))
            continue
        plotter = LogsPlotter(config=importer.config, logs=importer.checkpoint["logs"])

        importer.fold = count*10 + fold
        # try:
        #     importer.change_config(attr="model.args.shared_preds", value=False)
        #     a = importer.get_model()
        #     importer._my_numel(a)
        # except:
        #     print("We could not load {}".format(config_name))
        #     continue

        multi_fold_results = importer.print_progress(multi_fold_results=multi_fold_results, latex_version=False)
        importer.print_progress_aggregated(multi_fold_results=multi_fold_results, latex_version=True)



        # if (i+1)%3 == 0:
        #     print(multi_fold_results.keys())
        #     eeg.append(np.array([multi_fold_results[i]["k"]["eeg"] for i in multi_fold_results]).mean())
        #     eog.append(np.array([multi_fold_results[i]["k"]["eog"] for i in multi_fold_results]).mean())
        #     mm.append(np.array([multi_fold_results[i]["k"]["combined"] for i in multi_fold_results]).mean())
        #     eeg_std.append(np.array([multi_fold_results[i]["k"]["eeg"] for i in multi_fold_results]).std())
        #     eog_std.append(np.array([multi_fold_results[i]["k"]["eog"] for i in multi_fold_results]).std())
        #     mm_std.append(np.array([multi_fold_results[i]["k"]["combined"] for i in multi_fold_results]).std())
        #     multi_fold_results = {}

        # plotter.sleep_plot_losses()
        # plotter.sleep_plot_performance()
    count +=1

# eeg = np.array([multi_fold_results[i]["k"]["eeg"] for i in multi_fold_results])
# eog = np.array([multi_fold_results[i]["k"]["eog"] for i in multi_fold_results])
# mm = np.array([multi_fold_results[i]["k"]["combined"] for i in multi_fold_results])
# x = np.array([ 10, 20, 40, 60, 80])
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,6))
#
# ax = plt.subplot(111)
# plt.plot(x, eeg, color="blue", label="EEG")
#
# plt.plot(x, eog, color="orange", label="EOG")
# plt.plot(x, mm, color="black", label="Multimodal")
# plt.legend(fontsize=12, frameon=False, ncol=3,  bbox_to_anchor=(0.28, 0.88))
# plt.xlabel("Outer Sequence Length", fontsize=12)
# plt.ylabel("Cohens Kappa", fontsize=12)
# plt.xticks(x, fontsize = 8)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# # ax.spines['bottom'].set_visible(False)
# plt.yticks([0.75,0.80,0.85], fontsize = 8)
#
# plt.xlim(0,100)
# plt.ylim(0.75,0.9)
# plt.title("Training with different outer sequence", fontsize=14)

# plt.show()

# eeg, eog, mm = np.array(eeg), np.array(eog), np.array(mm)
# eeg_std, eog_std, mm_std = np.array(eeg_std), np.array(eog_std), np.array(mm_std)
# x = np.array([  0, 50,  100,  200, 400,  600, 1000, 1500, 2000])
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,6))
# ax = plt.subplot(111)
# plt.plot(x, eeg, color="blue", label="EEG")
# plt.plot(x, eog, color="orange", label="EOG")
# plt.plot(x, mm, color="black", label="Multimodal")
# plt.scatter(x, eeg, color="blue")
# plt.scatter(x, eog, color="orange")
# plt.scatter(x, mm, color="black")
# plt.fill_between(x, eeg-eeg_std, eeg+eeg_std, alpha=0.5)
# plt.fill_between(x, eog-eog_std, eog+eog_std, alpha=0.5)
# plt.fill_between(x, mm-mm_std, mm+mm_std, alpha=0.5)
# plt.legend(fontsize=12, frameon=False, ncol=3,  bbox_to_anchor=(0.28, 0.88))
# # plt.xlabel("Extra Single-Modality EEG patients", fontsize=12)
# plt.xlabel("Extra Single-Modality EOG patients", fontsize=12)
# # plt.xlabel("Extra Single-Modality Patients", fontsize=12)
# plt.ylabel("Cohens Kappa", fontsize=12)
# plt.xticks(x, fontsize = 8)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# plt.yticks([0.65, 0.70,0.75,0.80,0.85 ], fontsize = 8)
# plt.xlim(-100,2100)
# plt.xlim(-100,1600)
# plt.ylim(0.65,0.85)
# # plt.title("Training with 100 EEG-EOG patients and EEG unimodal ones", fontsize=14)
# plt.title("Training with 100 EEG-EOG patients and EOG unimodal ones", fontsize=14)
# # plt.title("Training with 100 EEG-EOG patients and unimodal ones", fontsize=14)
# # plt.ylim(0.75,0.9)
# plt.savefig("./incomplete_100c_2000e_0o.svg")
# plt.savefig("./incomplete_100c_2000e_0o.png")
# plt.show()

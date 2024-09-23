from enum import Enum


class TrecMetric(str, Enum):
    NUM_Q = "num_q"
    NUM_RET = "num_ret"
    NUM_REL = "num_rel"
    NUM_REL_RET = "num_rel_ret"
    # Mean average precision
    MAP = "map"
    GM_MAP = "gm_map"
    R_PREC = "Rprec"
    B_PREF = "bpref"
    RECIP_RANK = "recip_rank"

    IPREC_AT_RECALL_0 = "iprec_at_recall_0.00"
    IPREC_AT_RECALL_01 = "iprec_at_recall_0.10"
    IPREC_AT_RECALL_02 = "iprec_at_recall_0.20"
    IPREC_AT_RECALL_03 = "iprec_at_recall_0.30"
    IPREC_AT_RECALL_04 = "iprec_at_recall_0.40"
    IPREC_AT_RECALL_05 = "iprec_at_recall_0.50"
    IPREC_AT_RECALL_06 = "iprec_at_recall_0.60"
    IPREC_AT_RECALL_07 = "iprec_at_recall_0.70"
    IPREC_AT_RECALL_08 = "iprec_at_recall_0.80"
    IPREC_AT_RECALL_09 = "iprec_at_recall_0.90"
    IPREC_AT_RECALL_1 = "iprec_at_recall_1.00"

    # Precision at different ranks
    P_5 = "P_5"
    P_10 = "P_10"
    P_15 = "P_15"
    P_20 = "P_20"
    P_30 = "P_30"
    P_100 = "P_100"
    P_200 = "P_200"
    P_500 = "P_500"
    P_1000 = "P_1000"

    # Recall at different ranks
    R_5 = "recall_5"
    R_10 = "recall_10"
    R_15 = "recall_15"
    R_20 = "recall_20"
    R_30 = "recall_30"
    R_100 = "recall_100"
    R_200 = "recall_200"
    R_500 = "recall_500"
    R_1000 = "recall_1000"

    # Accuracy @ 1
    A_1 = "success_1"
    A_5 = "success_5"
    A_10 = "success_10"

    # Normalized Discounted Cumulative Gain
    NDCG = "ndcg"

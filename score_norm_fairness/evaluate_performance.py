#!/usr/bin/env python3
import sys
sys.path.append("..")
import numpy
import pandas as pd
import os
import metrics_only
from collections import defaultdict
from sklearn.metrics import roc_curve, auc


def preprocess_scores(directory):
    df = pd.read_csv(directory)
    df['label'] = (df['probe_subject_id'] == df['bio_ref_subject_id']).astype(int)
    df['similarity'] = df['score']
    df['race'] = df['probe_race']
    return df
        
def load_split(df):
    genuines = df[df['label'] == 1]
    impostors = df[df['label'] == 0]
    return genuines, impostors

def main(score_file):
    # Example
    fmr_thresholds=[0.001,0.01,0.1]
    stats_type = ["Max-Min", "Max/Min", "Max/GeoMean", "Gini"]

    # Split the scores(similarity) into positives and negatives
    df = preprocess_scores(score_file)
    positives, negatives = load_split(df)

    # Further split scores(similarity) by race label
    negatives_as_dict, positives_as_dict = metrics_only.split_scores_by_variable(
        negatives, positives, "race"
    )

    # compute decision threshold give the FMR threshold
    taus = [metrics_only.compute_fmr_thresholds(negatives["similarity"].to_numpy(), [threshold]) for threshold in fmr_thresholds if fmr_thresholds is not None]

    # Compute FMR and FNMR at the specific decision threshold (far is the same as fmr, while frr is the same as fnmr, just different names)
    false_positive, false_negative, fmr, fnmr = metrics_only.farfrr(negatives["similarity"].to_numpy(), positives["similarity"].to_numpy(), taus[0])
    print(f"\nWhen fmr=0.1%, the threshold is {taus[0]}.....")
    print(f'fmr: {fmr}')
    print(f'fnmr: {fnmr}')
    # TMR = 1 - FNMR
    print(f"tmr: {1-fnmr}")

    # Compute EER
    eer, eer_threshold = metrics_only.eer(negatives["similarity"].to_numpy(), positives["similarity"].to_numpy())
    print(f"eer: {eer * 100:.2f}% \n")

    r_accuracy = []
    for key in negatives_as_dict.keys():
        r_neg_race = negatives_as_dict[key]["similarity"].to_numpy()
        r_pos_race = positives_as_dict[key]["similarity"].to_numpy()
        # Compute FMR-threshold for specific race
        r_taus = [metrics_only.compute_fmr_thresholds(r_neg_race, [threshold]) for threshold in fmr_thresholds if fmr_thresholds is not None]
        r_fp, r_fn, r_fmr, r_fnmr = metrics_only.farfrr(r_neg_race, r_pos_race, r_taus[0])
        print(f"[{key}] tmr: {(1-r_fnmr) * 100:.2f}%")
        r_eer, r_eer_threshold = metrics_only.eer(r_neg_race, r_pos_race)
        print(f"[{key}] eer: {r_eer * 100:.2f}%")
        r_df = df[df['race'] == key]
        r_acc, r_acc_threshold = metrics_only.compute_accuracy(r_df['similarity'].to_numpy(), r_df['label'].to_numpy())
        r_accuracy.append(r_acc * 100)
        print(f"[{key}] Accuracy: {r_acc * 100:.2f}% \n")

    avg_acc = numpy.mean(r_accuracy)
    std_acc = numpy.std(r_accuracy)

    print(f"Overall Accuracy: {avg_acc:.4f}")
    print(f"STD of Accuracy: {std_acc:.4f}")

    for state in stats_type:
        print(f'\nstate: {state}------------------------')
        fmrs = []
        fnmrs = []
        fmrs_all_adjusted = []
        fnmrs_all_adjusted = []
        fmrs_zero_adjusted = []
        fnmrs_zero_adjusted = []

        # compute demo-specific fmr and fnmr
        # FPD = False Positive Disparity (Measures disparity in FMR across groups)
        # FND = False Negative Disparity (Measures disparity in FNMR across groups)
        for key in negatives_as_dict.keys():

            neg_race = negatives_as_dict[key]["similarity"].to_numpy()
            pos_race = positives_as_dict[key]["similarity"].to_numpy()
            fp, fn, fmr, fnmr = metrics_only.farfrr(neg_race, pos_race, taus[0])
            
            fmrs.append(fmr)
            fnmrs.append(fnmr)

            # if only zero adjusted and keep non-zero error rates, then replace zero error rate in the denominator by the adjusted version 
            if "Max/GeoMean" == state:
                fmrs_zero_adjusted.append(metrics_only.zero_error_rate(fp,len(neg_race)))
                fnmrs_zero_adjusted.append(metrics_only.zero_error_rate(fn,len(pos_race)))

        if "Max-Min" == state: #eq 1, eq2
            term_A, term_B = metrics_only.compute_differential_error(A_tau=numpy.array(fmrs),B_tau=numpy.array(fnmrs))
            print(f"FMR: {term_A:.4f}")
            print(f"FNMR: {term_B:.4f}")
        
        if "Max/Min" == state: #eq 3, eq4, eq5
            term_A, term_B = metrics_only.compute_ser(A_tau=numpy.array(fmrs),B_tau=numpy.array(fnmrs))
            print(f"FMR: {term_A:.4f}")
            print(f"FNMR: {term_B:.4f}")

        if "Max/GeoMean" == state: #eq 6, eq7, eq9, section 4.2
            term_A, term_B = metrics_only.compute_merg_adjusted(A_tau=[numpy.array(fmrs),numpy.array(fmrs_zero_adjusted)],B_tau=[numpy.array(fnmrs),numpy.array(fnmrs_zero_adjusted)],alpha=0.5, beta=0.5)
            print(f"FMR: {term_A:.4f}")
            print(f"FNMR: {term_B:.4f}")

        if "Gini" == state: #eq 13, eq14
            term_A, term_B = metrics_only.compute_gini(A_tau=numpy.array(fmrs),B_tau=numpy.array(fnmrs),alpha=0.5, beta=0.5)
            print(f"FMR: {term_A:.4f}")
            print(f"FNMR: {term_B:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {__name__} <score-file>")
        exit(0)

    score_file = sys.argv[1]
    main(score_file)

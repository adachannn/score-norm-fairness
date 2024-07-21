#!/usr/bin/env python3
import os
import pandas as pd 
from sklearn import metrics
from scipy import interpolate
from collections import defaultdict
import json

"""Generate the dictionary to record tmr values at each fmr threshold, for each given score files."""


def read_scores(directory):
    df = pd.read_csv(directory,delimiter=",")
    df["genuine"] = (df["probe_subject_id"] == df["bio_ref_subject_id"]).astype(int) - 1
    return df 

def compute_roc(df,fixed_thresholds=[1e-4,1e-3,1e-2,1e-1]):

    fmr, tmr, thresholds = metrics.roc_curve(df["genuine"], df["score"], pos_label=0)
    auc = metrics.auc(fmr, tmr)
    tmr_intrp = interpolate.interp1d(fmr, tmr)
    fixed_tmrs = []

    for thre in fixed_thresholds:
        fixed_tmrs.append(tmr_intrp(thre).item())

    return fixed_tmrs

def save_to_json(dictionary,output):

    with open(output,"w") as f:
        json.dump(dictionary, f)

def generate_dictionary(scores,titles,num_thresholds,output):
    thresholds = list(range(1, num_thresholds+1))
    thresholds.reverse()
    for i in range(len(thresholds)):
        thresholds[i] = 10**(-thresholds[i])
    all_tmrs = defaultdict(list)

    for score in scores:
        df = read_scores(score)
        fixed_tmrs = compute_roc(df,thresholds)
        all_tmrs[titles[scores.index(score)]] = fixed_tmrs
    
    with open(output,"w") as f:
        json.dump(all_tmrs, f)

def calculate_tmr_per_directory(args):

    scores = args.scores_directory.split(',')
    titles = args.titles.split(',')
    generate_dictionary(scores,titles,args.num_thresholds,args.output)

    

import argparse
def get_args(command_line_options = None):
    
    parser = argparse.ArgumentParser("ROC_json",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--scores_directory","-dir", default = None, type=str, help = "A comma-separated list of directories for csv scores files")
    parser.add_argument("--titles","-t", default = None, type=str, help = "A comma-separated list of titles correspond to the scores")
    parser.add_argument("--num_thresholds","-nt", default = 0, type=int, help = "number of thresholds")    
    parser.add_argument("--output","-o", default = None, type=str, help = "OUTPUT json file directory, summarize the tmr at all required fmr thresholds")

    args = parser.parse_args(command_line_options)

    return args


if __name__ == '__main__':
    args = get_args()
    calculate_tmr_per_directory(args)
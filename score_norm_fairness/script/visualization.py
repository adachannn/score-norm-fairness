
import json
import os
import pandas as pd
from collections import defaultdict
import argparse

def threshold_specific(fairness,tmr,num):
    with open(fairness,"r") as f:
        fairness = json.load(f)
    with open(tmr,"r") as f:
        tmr = json.load(f)

    score = defaultdict(lambda:defaultdict())
    baseline_tmr = tmr["raw"][-num]
    baseline_fair = fairness["raw"]["fairness"]
    baseline_fair = baseline_fair[-num]
    baseline_fmr = fairness["raw"]["fmrs_scaled"]
    baseline_fmr = baseline_fmr[-num]
    baseline_fnmr = fairness["raw"]["fnmrs_scaled"]
    baseline_fnmr = baseline_fnmr[-num]

    for key in fairness.keys():
        if key == "raw":
            score["Baseline"]["TMR"] = baseline_tmr*100
            score["Baseline"]["WERM"] = baseline_fair
            score["Baseline"]["M/M FMR"] = baseline_fmr**2
            score["Baseline"]["M/M FNMR"] = baseline_fnmr**2

        else:
            update_key = key
            score[update_key]["TMR"] = tmr[key][-num]*100
            score[update_key]["M/M FMR"] = fairness[key]["fmrs_scaled"][-num]**2
            score[update_key]["M/M FNMR"] = fairness[key]["fnmrs_scaled"][-num]**2
            fairness[key] = fairness[key]["fairness"][-num]
            score[update_key]["WERM"] = fairness[key]

    
    return score

def to_df(dictionary):
    
    # Flatten the nested dictionary into a list of dictionaries
    flattened_data = []
    for first_key, second_dict in dictionary.items():
        for second_key, third_dict in second_dict.items():
            for third_key, value in third_dict.items():
                flattened_data.append({
                    'Network': first_key,
                    'Method': second_key,
                    'Metrics': third_key,
                    'Value': value
                })
    
    # Create a DataFrame from the flattened data
    df = pd.DataFrame(flattened_data)

    # Pivot the DataFrame to rearrange columns
    pivot_df = df.pivot_table(index='Method', columns=['Network','Metrics'], values='Value')

    desired_column_order = ['TMR','WERM', 'M/M FMR', 'M/M FNMR']
    # Reindex the columns of pivot_df based on the desired order
    pivot_df = pivot_df.reindex(columns=desired_column_order, level=1)
    pivot_df.index.name = None
    pivot_df = pivot_df.round(4)

    return pivot_df

def save_to_csv(out,output):

    out.to_csv(output, header=True, index=True, sep=',', mode='w')

def fair_score_analysis(score_path,demo,extractors,output,target_threshold=3):

    analysis = dict()
    for extractor in extractors:
        werm_report = os.path.join(score_path,f"{demo}_{extractor}_report.json")
        tmr_score = os.path.join(score_path,f"{demo}_{extractor}_table.json")
        result = threshold_specific(werm_report,tmr_score,target_threshold)
        analysis[extractor] = result
    

    analysis_df = to_df(analysis)

    save_to_csv(analysis_df,output)
    

def get_args(command_line_options = None):

    parser = argparse.ArgumentParser("visualization",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--score-directory","-s", required=True, help = "Score directory for a specific dataset (and protocol)")
    parser.add_argument("--demo-name","-n", choices = ("race", "gender"), default = "race", type=str, help = "Specific demographic to be used for normalization, possible choices: race, gender")
    parser.add_argument("--extractors","-e", nargs='+', choices = ("E1", "E2", "E3", "E4", "E5"),  default = ["E1"], help = "The list of feature extractors")
    parser.add_argument("--target-thresholds","-T", default = 3, type=int, help = "The target FMR threshold 10^-T")
    parser.add_argument("--output","-o", default="analysis_table.csv", help = "OUTPUT csv file path to save the summarization table")

    args = parser.parse_args(command_line_options)

    return args


def main():
    args = get_args()
    fair_score_analysis(args.score_directory,args.demo_name,args.extractors,args.output,args.target_thresholds)

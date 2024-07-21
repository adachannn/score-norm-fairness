#!/usr/bin/env python3
import sys
sys.path.append("..")
import os
from postprocess import normalize
from score import vgg_score, rfw_score

def pipeline(args):

    stages = args.stage.split(',')
    methods = args.methods.split(',')

    for method in methods:
        if method in ["M3","M4","M5"]:
            same_demo_train = same_demo_test = args.demo_name
            same_race = True
            compare_type = ("cohort","cohort")
            train_sample = 0
        elif method == "M1":
            compare_type = ("test","cohort")
            same_race = False
            same_demo_train = None
            same_demo_test = args.demo_name
            train_sample = 200
        elif method == "M1.1":
            compare_type = ("test","cohort") 
            same_race = False
            same_demo_train = None
            same_demo_test = args.demo_name
            train_sample = 1000
        elif method == "M1.2":
            compare_type = ("test","cohort")
            same_race = False
            same_demo_train = args.demo_name
            same_demo_test = args.demo_name
            train_sample = 25
        elif method == "M2":
            compare_type = ("cohort","test")
            same_race = False
            same_demo_train = None
            same_demo_test = args.demo_name
            train_sample = 200
        elif method == "M2.1":
            compare_type = ("cohort","test") 
            same_race = False
            same_demo_train = None
            same_demo_test = args.demo_name
            train_sample = 1000
        elif method == "M2.2":
            compare_type = ("cohort","test")
            same_race = True
            same_demo_train = args.demo_name
            same_demo_test = args.demo_name
            train_sample = 25
        
        if "train" in stages:
            if args.dataset == "rfw":
                rfw_score(args.data_directory,args.protocol_directory,args.protocol,train_sample,same_race=same_race,output=args.output_directory,file_name=f"{method}_cohort",compare_type=compare_type)

            elif args.dataset == "vgg2":
                vgg_score(args.data_directory,args.protocol_directory,args.protocol,same_demo_test=same_demo_test,same_demo_train=same_demo_train,output=args.output_directory,file_name=f"{method}_cohort",compare_type=compare_type)
        if "test" in stages:
            raw_scores = os.path.join(args.output_directory,"raw.csv")
            if args.dataset == "rfw":
                demo_groups = ["African","Asian","Caucasian","Indian"]
            elif args.dataset == "vgg2":
                if args.demo_name == "race": demo_groups = ["A","B","I","W"]
                elif args.demo_name == "gender": demo_groups = ["m","f"]
                
            cohort_scores = os.path.join(args.output_directory,f"{method}_cohort.csv")
            normalize(raw_scores,cohort_scores,args.demo_name,demo_groups,method,args.output_directory)




import argparse
def get_args(command_line_options = None):
    
    parser = argparse.ArgumentParser("Score-Norm-Pipeline",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--stage","-stg", default = "train,test", type=str, help = "A comma-separated list of stages, possible choices: train, test. If train, generate raw scores and cohort scores; if test, use cohort scores to compute statistics to normalize raw scores.")
    parser.add_argument("--methods","-m", default = "M1", type=str, help = "A comma-separated list of score normalization methods to be applied, possible choices: M1, M1.1, M1.2, M2, M2.1, M2.2, M3, M4, M5")
    parser.add_argument("--demo_name","-demo", default = "race", type=str, help = "Specific demographic to be used for normalization, possible choices: race, gender")   
    parser.add_argument("--dataset","-d", default = "rfw", type=str, help = "Dataset, possible choices: rfw, vgg2") 
    parser.add_argument("--protocol","-p", default = "original", type=str, help = "Specify the protocol for dataset, possible choices for rfw: original, random; possible choices for vgg2: vgg2-short-demo, vgg2-full-demo") 
    parser.add_argument("--data_directory","-dr", default = None, type=str, help = "Dataset/Image directory") 
    parser.add_argument("--protocol_directory","-pr", default = None, type=str, help = "Protocol directory") 
    parser.add_argument("--output_directory","-o", default = None, type=str, help = "OUTPUT directory, all csv scores files, including raw scores, cohort scores, and normalized scores.")

    args = parser.parse_args(command_line_options)

    return args


if __name__ == '__main__':
    args = get_args()
    pipeline(args)
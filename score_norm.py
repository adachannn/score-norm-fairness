#!/usr/bin/env python3
import argparse
import sys
sys.path.append("..")
import os
from postprocess import normalize
from score import vgg_score, rfw_score

def pipeline(args):

    for method in args.methods:
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

        if "train" in args.stages:
            if args.dataset == "rfw":
                rfw_score(args.data_directory,args.protocol_directory,args.protocol,train_sample,same_race=same_race,output=args.output_directory,file_name=f"{method}_cohort",compare_type=compare_type)

            elif args.dataset == "vgg2":
                vgg_score(args.data_directory,args.protocol_directory,args.protocol,same_demo_test=same_demo_test,same_demo_train=same_demo_train,output=args.output_directory,file_name=f"{method}_cohort",compare_type=compare_type)

        if "test" in args.stages:
            raw_scores = os.path.join(args.output_directory,"raw.csv")
            if args.dataset == "rfw":
                demo_groups = ["African","Asian","Caucasian","Indian"]
            elif args.dataset == "vgg2":
                if args.demo_name == "race": demo_groups = ["A","B","I","W"]
                elif args.demo_name == "gender": demo_groups = ["m","f"]

            cohort_scores = os.path.join(args.output_directory,f"{method}_cohort.csv")
            normalize(raw_scores,cohort_scores,args.demo_name,demo_groups,method,args.output_directory)


def get_args(command_line_options = None):

    parser = argparse.ArgumentParser("Score-Norm-Pipeline",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--stages","-s", nargs='+', choices=("train", "test"), default = ["train","test"], help = "A list of stages, possible choices: train, test. If train, generate raw scores and cohort scores; if test, use cohort scores to compute statistics to normalize raw scores.")
    parser.add_argument("--methods","-m", nargs='+', choices = ("M1", "M1.1", "M1.2", "M2", "M2.1", "M2.2", "M3", "M4", "M5"),  default = ["M1"], help = "A comma-separated list of score normalization methods to be applied, possible choices: M1, M1.1, M1.2, M2, M2.1, M2.2, M3, M4, M5")
    parser.add_argument("--demo-name","-n", choices = ("race", "gender"), default = "race", type=str, help = "Specific demographic to be used for normalization, possible choices: race, gender")
    parser.add_argument("--dataset","-d", choices = ("ref", "vgg2"), default = "rfw", type=str, help = "Dataset, possible choices: rfw, vgg2")
    parser.add_argument("--protocol","-p", choices = ("original", "random", "vgg2-short-demo", "vgg2-full-demo"), default = "original", type=str, help = "Specify the protocol for dataset, possible choices for rfw: original, random; possible choices for vgg2: vgg2-short-demo, vgg2-full-demo")
    parser.add_argument("--data-directory","-D", help = "Dataset/Image directory")
    parser.add_argument("--protocol-directory","-P", help = "Protocol directory")
    parser.add_argument("--output-directory","-o", help = "OUTPUT directory, all csv scores files, including raw scores, cohort scores, and normalized scores.")

    args = parser.parse_args(command_line_options)

    return args


if __name__ == '__main__':
    args = get_args()
    pipeline(args)

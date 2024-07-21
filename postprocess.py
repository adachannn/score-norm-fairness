import os
import pandas as pd
import numpy
from scipy.stats import norm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def read_scores(directory):
    
    df = pd.read_table(directory, delimiter=",")

    return df

def save(df,output):
    
    df.set_index(df.iloc[:, 0],inplace=True)
    df = df.drop(columns=["probe_key"])
    df.to_csv(output,header=True, index=True, sep=",", mode='w')

def compute_stats(raw,cohort,demo_name,demo,method):
    """
    Compute statistics and normalize raw scores

    Parameters
    ----------
      raw: dataframe
        Pandas Dataframe containing the raw scores

      cohort: dataframe
        Pandas Dataframe containing the cohort scores used to compute statistics

      demo_name: str
        Demographic type used to split scores by demographics

      demo: list
        List of strings, i.e. list of demographic groups within the "demo_name"

      method: str
        Specify the score normalization method applied
    """

    if method in ["M1.2","M2.2","M3","M4","M5"]: ## methods that require cohort scores all from same demogrphic groups

        post_scores = pd.DataFrame()
        ## iterate for all demographic groups, compute statistic within each group and normalize scores with gallery subject from this specific group, regardless of the demographic group for probe subject
        for r in demo:
            
            # df = cohort[(cohort[f"probe_{demo_name}"] == cohort[f"bio_ref_{demo_name}"]) & (cohort[f"bio_ref_{demo_name}"] == r)]
            # post_processed = raw[(raw[f"probe_{demo_name}"] == raw[f"bio_ref_{demo_name}"]) & (raw[f"bio_ref_{demo_name}"] == r)]

            ## Select cohort pair with gallery and probe from same demographic groups
            df = cohort[(cohort[f"probe_{demo_name}"] == cohort[f"bio_ref_{demo_name}"]) & (cohort[f"bio_ref_{demo_name}"] == r)]
            ## Select test pair with gallery subject from this demographic group
            post_processed = raw[raw[f"bio_ref_{demo_name}"] == r]

            ## Split cohort pairs into genuine and impostor for next step calculation
            genuine = df[df["probe_subject_id"] == df["bio_ref_subject_id"]]
            impostor = df[df["probe_subject_id"] != df["bio_ref_subject_id"]]


            if method == "M5": ## fit a genuine model and an impostor model with cohort scores

                genuine_mean = genuine["score"].mean()
                genuine_std = genuine["score"].std()
                impostor_mean = impostor["score"].mean()
                impostor_std = impostor["score"].std()

                post_processed["score"] = (
                    norm.cdf(post_processed["score"],genuine_mean,genuine_std) - 1. + norm.cdf(post_processed["score"],impostor_mean,impostor_std)
                )

            elif method == "M4": ## use platt scaling to normalize scores

                genuine_label = numpy.array(len(genuine) * [1.])
                impostor_label = numpy.array(len(impostor) * [0.])

                X = numpy.concatenate((genuine["score"].values,impostor["score"].values)).reshape(-1,1)
                y = numpy.concatenate((genuine_label,impostor_label))

                regressor = LogisticRegression(
                    class_weight="balanced", fit_intercept=True, penalty="l2"
                )
                regressor.fit(X, y)
                coef = regressor.coef_
                intercept = regressor.intercept_

                post_processed["score"] = expit(post_processed["score"]*coef[0][0] + intercept[0])

            elif method == "M1.2" or method == "M2.2" or method == "M3": ## for each demographic group, collect all impostor scores, compute its mean and std and normalize all scores within this demographic group; the sources for impostor scores are different, respectively

                impostor_mean = impostor["score"].mean()
                impostor_std = impostor["score"].std()
                
                post_processed["score"] = (post_processed["score"] - impostor_mean) / impostor_std

            post_scores = pd.concat([post_scores,post_processed])
    
    else: ## methods that take cross demographic impostor scores

        post_scores = pd.DataFrame()
        
        ## znorm, i.e. M1/M1.1, group scores based on gallery sample's id/demographic, and then compute statistic and apply normalization; tnorm, i.e. M2/M2.1, use probe sample's id/demographic.
        id_criterion = "bio_ref" if method in ["M1","M1.1"] else "probe"
        demo_critierion = "probe" if method in ["M1","M1.1"] else "bio_ref"

        if method == "M1" or method == "M2": ## subject-based, so get all impostor scores per gallery/probe sample

            ## add missing column for groupby and merging purpose
            cohort["bio_ref_key"] = cohort["bio_ref_subject_id"] + "/" + cohort["bio_ref_reference_id"]

            group_stats = cohort.groupby(f'{id_criterion}_key')['score'].agg(['mean', 'std']).reset_index()

            raw["bio_ref_key"] = raw["bio_ref_subject_id"] + "/" + raw["bio_ref_reference_id"]
            raw = raw.merge(group_stats, on=f'{id_criterion}_key', how='left')
            raw['normalized_score'] = (raw['score'] - raw['mean']) / raw['std']

            # Drop unnecessary columns
            raw = raw.drop(columns=['mean', 'std','bio_ref_key','score'])
            raw.rename(columns={'normalized_score':'score'}, inplace=True)
            post_scores = raw
        
        elif method == "M1.1" or method == "M2.1": ## subject-demo-based, so get all impostor scores per gallery/probe sample per demographic group

            cohort["bio_ref_key"] = cohort["bio_ref_subject_id"] + "/" + cohort["bio_ref_reference_id"]
                
            group_stats = cohort.groupby([f'{id_criterion}_key', f'{demo_critierion}_{demo_name}'])['score'].agg(['mean', 'std']).reset_index()

            raw["bio_ref_key"] = raw["bio_ref_subject_id"] + "/" + raw["bio_ref_reference_id"]
            raw = raw.merge(group_stats, on=[f'{id_criterion}_key',f'{demo_critierion}_{demo_name}'], how='left')

            raw['normalized_score'] = (raw['score'] - raw['mean']) / raw['std']

            # Drop unnecessary columns
            raw = raw.drop(columns=['mean', 'std','bio_ref_key','score'])
            raw.rename(columns={'normalized_score':'score'}, inplace=True)
            post_scores = raw

    return post_scores

def normalize(raw,cohort,demo_name,demo,method,output):

    ## load raw scores and cohort scores
    raw = read_scores(raw)
    cohort = read_scores(cohort)

    ## compute statistics based on cohort scores and apply them to normalize raw scores
    post_scores = compute_stats(raw,cohort,demo_name,demo,method)
    save(post_scores,os.path.join(output,f"{method}_normed.csv"))

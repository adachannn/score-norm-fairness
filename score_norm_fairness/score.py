import pandas as pd
import os
import h5py
from .dataset import RFW, VGG2
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.distance import cdist

def load_features(sampleset):
    """
    Load features from .h5 files and append feature to the dictionary of each sample

    sampleset: List of dictionaries
    """
    sampleset_features = defaultdict(dict)
    for sample in sampleset:
        try:
            with h5py.File(sample["path"], 'r') as h5_file:
                assert len(h5_file) == 1
                sampleset_features[sample["key"]] = sample
                sampleset_features[sample["key"]]["feature"] = h5_file[next(iter(h5_file))][()]
        except Exception as e:
            print(f"Failed to load {sample['path']}: {e}")
    return sampleset_features



def compute_scores(probes,gallery,all_vs_all=False,same_demo_compare=None):
    """
    Compute the cosine similarity scores between probe samples and gallery samples

    Probe: List of dictionaries of probe samples

    Gallery: List of dictionaries of gallery samples

    all_vs_all: Bool, if True, ignore `references` of probe sample and compare all gallery samples with all probe samples; if False, compare the gallery samples listed in `references` of each probe sample

    same_demo_compare: str (None, race, gender), when all_vs_all is True, set same_demo_compare can ignore the cross demo pairs and speed up the computation when cross demo pairs are not necessary

    Return: dictionary of scores
    """
    scores = defaultdict(dict)
    if not all_vs_all:
        for sample in tqdm(probes):
            for ref in probes[sample]["references"]:
                # score = 2 - scipy.spatial.distance.cosine(probes[sample]["feature"],gallery[ref]["feature"])
                score = -1 * cdist(probes[sample]["feature"].reshape(1, -1),gallery[ref]["feature"].reshape(1, -1), "cosine")

                key = (probes[sample]["key"],ref)
                scores[key]["probe_race"] = probes[sample]["race"]
                scores[key]["probe_gender"] = probes[sample]["gender"]
                scores[key]["bio_ref_race"] = gallery[ref]["race"]
                scores[key]["bio_ref_gender"] = gallery[ref]["gender"]
                scores[key]["score"] = score[0][0]
    else:

        ## group the reference keys based on their demographic
        if same_demo_compare:
            reference_lists = dict()
            for ref in gallery:
                if gallery[ref][same_demo_compare] not in reference_lists:
                    reference_lists[gallery[ref][same_demo_compare]] = []
                reference_lists[gallery[ref][same_demo_compare]].append(ref)

        for sample in tqdm(probes):
            if same_demo_compare:
                gallery_samples = reference_lists[probes[sample][same_demo_compare]]
            else: ## if cross demo pairs are necessary, then take the entire gallery
                gallery_samples = gallery
            for ref in gallery_samples:
                score = -1 * cdist(probes[sample]["feature"].reshape(1, -1),gallery[ref]["feature"].reshape(1, -1), "cosine")

                key = (probes[sample]["key"],ref)
                scores[key]["probe_race"] = probes[sample]["race"]
                scores[key]["probe_gender"] = probes[sample]["gender"]
                scores[key]["bio_ref_race"] = gallery[ref]["race"]
                scores[key]["bio_ref_gender"] = gallery[ref]["gender"]
                scores[key]["score"] = score[0][0]
    return scores


def save(scores,output):
    """
    Convert the scores in to a dataframe format and save to a csv file

    scores: dictionary of scores

    output: path to save the csv file
    """

    df = pd.DataFrame.from_dict(scores,orient="index")
    df = df.reset_index()
    df.rename(columns={'level_0': 'probe_key', 'level_1': 'bio_ref'}, inplace=True)

    split_columns = df['probe_key'].str.split('/', expand=True)
    names = split_columns.columns.tolist()
    split_columns.rename(columns={names[-1]: 'probe_reference_id', names[-2]: 'probe_subject_id'}, inplace=True)
    df[['probe_reference_id', 'probe_subject_id']] = split_columns[['probe_reference_id', 'probe_subject_id']]

    split_columns = df['bio_ref'].str.split('/', expand=True)
    names = split_columns.columns.tolist()
    split_columns.rename(columns={names[-1]: 'bio_ref_reference_id', names[-2]: 'bio_ref_subject_id'}, inplace=True)
    df[['bio_ref_reference_id', 'bio_ref_subject_id']] = split_columns[['bio_ref_reference_id', 'bio_ref_subject_id']]

    df.drop(columns=['bio_ref'], inplace=True)

    df = df.reindex(columns=["probe_key","probe_reference_id","probe_subject_id","probe_race","probe_gender","bio_ref_reference_id","bio_ref_subject_id","bio_ref_race","bio_ref_gender","score"])

    df.to_csv(output,header=True,index=False)


def standard_score(data,raw_file,file_name,compare_type=("cohort","test"),same_demo_compare=None):

    probe = data.probes()
    ref = data.references()
    probe_features = load_features(probe)
    ref_features = load_features(ref)

    if not os.path.isfile(raw_file):
        print("Computing raw scores")
        raw_scores = compute_scores(probe_features,ref_features)
        save(raw_scores,raw_file)

    print("Computing cohort scores")
    if compare_type == ("cohort","test"):
        tref = data.treferences()
        tref_features = load_features(tref)
        all_vs_all = True
        norm_scores = compute_scores(probe_features,tref_features,all_vs_all,same_demo_compare)
    elif compare_type == ("test","cohort"):
        zprobe = data.zprobes()
        zprobe_features = load_features(zprobe)
        all_vs_all = False
        norm_scores = compute_scores(zprobe_features,ref_features,all_vs_all,same_demo_compare)
    elif compare_type == ("cohort","cohort"):
        zprobe, tref = data.cohort_based_trainsamples()
        zprobe_features = load_features(zprobe)
        tref_features = load_features(tref)
        all_vs_all = False
        norm_scores = compute_scores(zprobe_features,tref_features,all_vs_all,same_demo_compare)

    save(norm_scores,file_name)


def rfw_score(data_directory,protocol_directory,protocol,train_sample,same_race,raw_file=None,file_name=None,compare_type=("cohort","test")):

    data = RFW(data_directory=data_directory,protocol_directory=protocol_directory,protocol=protocol,data_type="feature", train_sample=train_sample, same_race=same_race)
    if same_race: same_demo_compare = "race"
    else: same_demo_compare = None
    standard_score(
        data=data,
        raw_file=raw_file,
        file_name=file_name,
        compare_type=compare_type,
        same_demo_compare=same_demo_compare
    )


def vgg_score(data_directory,protocol_directory,protocol,same_demo_test=None,same_demo_train=None,raw_file=None,file_name=None,compare_type=("cohort","test")):

    data = VGG2(data_directory=data_directory,protocol_directory=protocol_directory,data_type="feature", protocol=protocol,same_demo_test=same_demo_test, same_demo_train=same_demo_train)
    standard_score(
        data=data,
        raw_file=raw_file,
        file_name=file_name,
        compare_type=compare_type,
        same_demo_compare=same_demo_train
    )

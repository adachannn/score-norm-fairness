import os
import copy
import numpy as np
import torch
from collections import defaultdict

class RFW(torch.utils.data.Dataset):
    """
    This is an extension version of Racial faces in the wild (RFW) dataset implementation defined in bob.bio.face, which proposed by Idiap Research Institute.

    The RFW is a subset of the MS-Celeb 1M dataset, and it's composed of 44332 images split into 11416 identities.
    There are four "race" labels in this dataset (`African`, `Asian`, `Caucasian`, and `Indian`).

    This interface considers two evaluation protocols.
    The first one, called "original", is the original protocol from its publication. It contains ~24k comparisons in total.
    The second one, called "random", contains the same genuine pairs as original protocol, but randomly selected impostor pairs with same race and gender.

    .. warning::
        The following identities are associated with two races in the original dataset
         - m.023915
         - m.0z08d8y
         - m.0bk56n
         - m.04f4wpb
         - m.0gc2xf9
         - m.08dyjb
         - m.05y2fd
         - m.0gbz836
         - m.01pw5d
         - m.0cm83zb
         - m.02qmpkk
         - m.05xpnv


    For more information check:

    .. code-block:: latex

        @inproceedings{wang2019racial,
        title={Racial faces in the wild: Reducing racial bias by information maximization adaptation network},
        author={Wang, Mei and Deng, Weihong and Hu, Jiani and Tao, Xunqiang and Huang, Yaohai},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={692--702},
        year={2019}
        }

    """
    def __init__(
        self,
        data_directory,
        protocol_directory,
        protocol="original",
        data_type="image",
        train_sample=25,
        same_race=True,
    ):
        """
        data_directory: str, directory to images

        protocol_directory: str, directory to protocols

        protocol: str (original, random), protocol name

        data_type: str (image, feature)

        train_sample: int, number of cohort samples for z/t-norm per race

        same_race: bool, if True, references for zprobe samples are from same race, to speed up training
        """

        if data_directory is None or not os.path.exists(data_directory):
            raise ValueError(
                "Invalid or non existant `data_directory`: f{data_directory}"
            )
        if protocol_directory is None or not os.path.exists(protocol_directory):
            raise ValueError(
                "Invalid or non existant `protocol_directory`: f{protocol_directory}"
            )

        self._check_protocol(protocol)
        self._races = ["African", "Asian", "Caucasian", "Indian"]

        self.data_directory = data_directory
        self.protocol_directory = protocol_directory
        self.protocol = protocol
        self.data_type = data_type
        self.train_sample = train_sample

        self.extension = ".h5" if self.data_type == "feature" else ".jpg"

        self._pairs = dict()
        self._pairs_subject = dict()
        self._inverted_pairs = dict()
        self._id_race = dict()  # ID -- > RACE
        self._race_ids = dict()  # RACE --> ID
        self._landmarks = dict()
        self._cached_biometric_references = None
        self._cached_probes = None
        self._cached_zprobes = None
        self._cached_coVSco_zprobes = None
        self._cached_treferences = None
        self._cached_coVSco_references = None
        self._discarded_subjects = (
            []
        )  # Some subjects were labeled with both races
        self._load_testdata()

        # Setting the seed so we have a consisent set of samples
        self._protocol_seed = 652

        # Set true to select only same race pairs for znorm, to save score calculation time
        self.same_race = same_race

    def _get_subject_from_key(self, key):
        return key[:-5]

    def _load_testdata(self):
        for race in self._races:
            if self.protocol == "random":
                final_destination = f"{race}_pairs_random.txt"
            elif self.protocol == "original":
                final_destination = f"{race}_pairs.txt"


            pair_file = os.path.join(
                self.protocol_directory,
                "test",
                "txts",
                race,
                final_destination,
            )

            for line in open(pair_file).readlines():
                line = line.split("\t")
                line[-1] = line[-1].rstrip("\n")

                key = f"{line[0]}_000{line[1]}"
                subject_id = self._get_subject_from_key(key)
                dict_key = f"{race}/{subject_id}/{key}"

                if subject_id not in self._id_race:
                    self._id_race[subject_id] = race
                else:
                    if (
                        self._id_race[subject_id] != race
                        and subject_id not in self._discarded_subjects
                    ):
                        logger.warning(
                            f"{subject_id} was already labeled as {self._id_race[subject_id]}, and it's illogical to be relabeled as {race}. "
                            f"This seems a problem with RFW dataset, so we are removing all samples linking {subject_id} as {race}"
                        )
                        self._discarded_subjects.append(subject_id)
                        continue

                # Positive or negative pairs
                if len(line) == 3:
                    k_value = f"{line[0]}_000{line[2]}"
                    dict_value = f"{race}/{self._get_subject_from_key(k_value)}/{k_value}"

                else:
                    k_value = f"{line[2]}_000{line[3]}"
                    dict_value = f"{race}/{self._get_subject_from_key(k_value)}/{k_value}"


                if dict_key not in self._pairs:
                    self._pairs[dict_key] = []
                self._pairs[dict_key].append(dict_value)

        # Preparing the probes
        self._inverted_pairs = self._invert_dict(self._pairs)
        self._race_ids = self._invert_dict(self._id_race)

    def _invert_dict(self, dict_pairs):
        inverted_pairs = dict()

        for k in dict_pairs:
            if isinstance(dict_pairs[k], list):
                for v in dict_pairs[k]:
                    if v not in inverted_pairs:
                        inverted_pairs[v] = []
                    inverted_pairs[v].append(k)
            else:
                v = dict_pairs[k]
                if v not in inverted_pairs:
                    inverted_pairs[v] = []
                inverted_pairs[v].append(k)
        return inverted_pairs

    def probes(self):

        if self._cached_probes is None:

            self._cached_probes = []
            for pair in self._inverted_pairs:
                sset = self._make_sample(pair)

                sset["references"] = [
                    key for key in self._inverted_pairs[pair]
                ]
                self._cached_probes.append(sset)

        return self._cached_probes

    def references(self):
        if self._cached_biometric_references is None:
            self._cached_biometric_references = []
            for key in self._pairs:
                self._cached_biometric_references.append(
                    self._make_sample(key)
                )
        return self._cached_biometric_references

    def _make_sample(self, item, target_set="test"):

        race, subject_id, reference_id = item.split("/")

        key = f"{race}/{subject_id}/{reference_id}"

        if target_set == "train":
            reference_id = f"{subject_id}/{reference_id}"
            annotations = self._fetch_landmarks(
                os.path.join(
                    self.protocol_directory,
                    "train/protocol/all_annotations.csv",
                ),
                reference_id,
                target_set,
            )
            next_path = "norm/" + key
        else:
            annotations = self._fetch_landmarks(
                os.path.join(
                    self.protocol_directory,
                    f"{target_set}/txts/{race}/{race}_lmk.txt",
                ),
                reference_id,
                target_set,
            )
            next_path = key

        if self.extension == ".jpg":
            path = (
                os.path.join(
                    self.data_directory,
                    f"{target_set}/data/{race}",
                    subject_id,
                    reference_id + self.extension,
                )
            )
        elif self.extension == ".h5":
            path = (
                os.path.join(
                    self.data_directory,
                    next_path + self.extension,
                )
            )


        sample = {"path":path,"key":key,"annotation":annotations,"reference_id":reference_id,"subject_id":subject_id,"race":race,"gender":"","annotation":annotations}

        return sample


    def zprobes(self):
        if self._cached_zprobes is None:
            self._cached_zprobes = self._load_traindata(
                self._protocol_seed + 1, "for_probes"
            )

            if self.same_race:
                reference_list = defaultdict(list)
                for race in self._races:
                    ref = list(
                        set([s["key"] for s in self.references() if s["race"] == race])
                    )
                    reference_list[race] = ref

                for p in self._cached_zprobes:
                    p["references"] = copy.deepcopy(reference_list[p["race"]])

            else:
                references = list(
                    set([s["key"] for s in self.references()])
                )
                for p in self._cached_zprobes:
                    p["references"] = copy.deepcopy(references)

        return self._cached_zprobes

    def treferences(self):
        if self._cached_treferences is None:
            self._cached_treferences = self._load_traindata(
                self._protocol_seed + 2, "for_models"
            )
        return self._cached_treferences

    def cohort_based_trainsamples(self):
        if self._cached_coVSco_zprobes is None or self._cached_coVSco_references is None:
            self._cached_coVSco_zprobes, self._cached_coVSco_references = self._load_traindata(csv_type=None)

        return self._cached_coVSco_zprobes, self._cached_coVSco_references

    def _load_traindata(self,seed=365,csv_type=None):

        if csv_type is not None:
            cache = []

            # Setting the seed so we have a consisent set of samples
            np.random.seed(seed)

            protocol_dir = os.path.join(
                    self.protocol_directory, "train", "protocol", csv_type+".csv"
            )
            subject_id_by_race = defaultdict(list)
            with open(protocol_dir) as f:
                next(f)
                for line in f.readlines():
                    line = line.split("\t")
                    subject_id = line[1]
                    race = line[0].split("/")[2]
                    if subject_id not in subject_id_by_race[race]:
                        subject_id_by_race[race].append(subject_id)

            for race in self._races:
                ids = subject_id_by_race[race]
                np.random.shuffle(ids)
                if self.train_sample == "all":
                    self.train_sample = len(ids)
                subject_ids = ids[0 : self.train_sample]

                selected = []

                with open(protocol_dir) as f:
                    next(f)
                    for line in f.readlines():
                        line = line.split("\t")
                        key = line[0].split("/")
                        key = "/".join(key[2:])
                        subject_id = line[1]
                        if (subject_id in subject_ids) and (selected.count(subject_id) < 1):
                            cache.append(
                                self._make_sample(
                                    key[:-4], target_set="train"
                                )
                            )
                            selected.append(subject_id)
            return cache
        else:
            if self._cached_coVSco_references is None:
                pairs = defaultdict(list)
                for race in self._races:

                    final_destination = f"{race}_pairs.txt"

                    protocol_dir = os.path.join(
                        self.protocol_directory, "train", "protocol", self.protocol, final_destination
                    )

                    with open(protocol_dir) as f:
                        for line in f.readlines():

                            id1, ref1, id2, ref2 = line.split("\t")
                            dict_key = f"{race}/{id1}/{ref1}"
                            dict_value = f"{race}/{id2}/{ref2.rstrip()}"
                            pairs[dict_key].append(dict_value)

                self._cached_coVSco_references = []
                for key in pairs:
                    self._cached_coVSco_references.append(
                        self._make_sample(key, target_set="train")
                    )
            if self._cached_coVSco_zprobes is None:
                self._cached_coVSco_zprobes = []
                invert_pairs = self._invert_dict(pairs)

                for pair in invert_pairs:
                    sset = self._make_sample(pair, target_set="train")
                    sset["references"] = [
                        key for key in invert_pairs[pair]
                    ]
                    self._cached_coVSco_zprobes.append(sset)
            return self._cached_coVSco_zprobes, self._cached_coVSco_references

    def _fetch_landmarks(self, filename, key, target_set="test"):
        if key not in self._landmarks:
            if target_set == "test":
                with open(filename) as f:
                    for line in f.readlines():
                        line = line.split("\t")
                        k = line[0].split("/")[-1][:-4]
                        self._landmarks[k] = dict()
                        splits = filename.split("/")
                        if ("train" in splits) and (("African" in splits) or ("Asian" in splits) or ("Indian" in splits)):
                            self._landmarks[k]["reye"] = (float(line[2]), float(line[1]))
                            self._landmarks[k]["leye"] = (float(line[4]), float(line[3]))
                            self._landmarks[k]["mouthright"] = (float(line[8]), float(line[7]))
                            self._landmarks[k]["mouthleft"] = (float(line[10]), float(line[9]))
                        else:
                            self._landmarks[k]["reye"] = (float(line[3]), float(line[2]))
                            self._landmarks[k]["leye"] = (float(line[5]), float(line[4]))
                            self._landmarks[k]["mouthright"] = (float(line[9]), float(line[8]))
                            self._landmarks[k]["mouthleft"] = (float(line[11]), float(line[10]))

            else:
                with open(filename) as f:
                    next(f)
                    for line in f.readlines():
                        line = line.split("\t")
                        kk = line[0].split("/")
                        r_k = kk[-1][:-4]
                        k = kk[-2] + "/" + r_k
                        self._landmarks[k] = dict()
                        splits = filename.split("/")
                        self._landmarks[k]["reye"] = (float(line[3]), float(line[2]))
                        self._landmarks[k]["leye"] = (float(line[5]), float(line[4]))
                        self._landmarks[k]["mouthright"] = (float(line[9]), float(line[8]))
                        self._landmarks[k]["mouthleft"] = (float(line[11]), float(line[10]))
        return self._landmarks[key]

    def __len__(self):
        return len(self.dataset["label"])

    def protocols(self):
        return ["original", "random"]

    def _check_protocol(self, protocol):
        assert (
            protocol in self.protocols()
        ), "Unvalid protocol `{}` not in {}".format(protocol, self.protocols())

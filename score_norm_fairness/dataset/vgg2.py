import os
import pandas as pd
import torch


class VGG2(torch.utils.data.Dataset):
    """
    This is an extension version of VGG2 implementation defined in bob.bio.face, which proposed by Idiap Research Institute.

    The VGG2 Dataset is composed of 9131 people split into two sets.
    The training set contains 8631 identities, while the test set contains 500 identities.

    As metadata, this dataset contains the gender labels "m" and "f" for, respectively, male and female.
    Further, we use the following race labels mentioned in bob.bio.face
    It includes:

        - A: Asian in general (Chinese, Japanese, Filipino, Korean, Polynesian, Indonesian, Samoan, or any other Pacific Islander
        - B: A person having origins in any of the black racial groups of Africa
        - I: American Indian, Asian Indian, Eskimo, or Alaskan native
        - U: Of indeterminable race
        - W: Caucasian, Mexican, Puerto Rican, Cuban, Central or South American, or other Spanish culture or origin, Regardless of race
        - N: None of the above
    Notice that we remove the identities with race label U and N, so in total 487 subjects.

    Relying on the available protocols from bob.bio.face, we develop two protocols `vgg2-short-demo`, `vgg2-full-demo`. Two protocols varie with respect to the number of samples per identity.

    The `vgg2-full-demo` preserves the number of samples per identity from the original dataset.
    On the other hand, the `vgg2-short-demo` presents 10 samples per identity at the probe and training sets. This is the one presents in our paper.

    All the landmarks and face crops provided in the original dataset is provided with this inteface.

    For more information check:

    .. code-block:: latex

        @inproceedings{cao2018vggface2,
            title={Vggface2: A dataset for recognising faces across pose and age},
            author={Cao, Qiong and Shen, Li and Xie, Weidi and Parkhi, Omkar M and Zisserman, Andrew},
            booktitle={2018 13th IEEE international conference on automatic face \\& gesture recognition (FG 2018)},
            pages={67--74},
            year={2018},
            organization={IEEE}
        }

    """

    def __init__(
        self,
        data_directory,
        protocol_directory,
        protocol="short-demo",
        data_type="image",
        same_demo_test=None,
        same_demo_train=None,
    ):

        """
        data_directory: str, directory to images

        protocol_directory: str, directory to protocols

        protocol: str (short-demo, full-demo), protocol name

        data_type: str (image, feature)

        same_race_test: str (None, race, gender), if True, references for probe samples are from same demo, to speed up training

        same_race_train: str (None, race, gender), if True, references for zprobe samples (cohort/training samples) are from same demo, to speed up training
        """
        if data_directory is None or not os.path.exists(data_directory):
            raise ValueError(
                f"Invalid or non existent `data_directory`: {data_directory}"
            )
        if protocol_directory is None or not os.path.exists(protocol_directory):
            raise ValueError(
                f"Invalid or non existent `protocol_directory`: {protocol_directory}"
            )

        # self._check_protocol(protocol)
        self._races = ["B", "A", "W", "I"]
        self._genders = ["M","F"]

        self.data_directory = data_directory
        self.protocol_directory = protocol_directory
        self.protocol = "vgg2-"+protocol
        self.data_type = data_type
        self.same_demo_test = same_demo_test
        self.same_demo_train = same_demo_train
        self.extension = ".h5" if self.data_type == "feature" else ".jpg"

        self._cached_biometric_references = None
        self._cached_probes = None
        self._cached_zprobes = None
        self._cached_treferences = None
        self._cached_coVSco_references = None
        self._cached_coVSco_zprobes = None

    def annotation(self,landmarks):

        lm = dict()

        lm["reye"] = (landmarks["REYE_Y"], landmarks["REYE_X"])
        lm["leye"] = (landmarks["LEYE_Y"], landmarks["LEYE_X"])
        lm["mouthright"] = (landmarks["RMOUTH_Y"], landmarks["RMOUTH_X"])
        lm["mouthleft"] = (landmarks["LMOUTH_Y"], landmarks["LMOUTH_X"])
        lm["nose"] = (landmarks["NOSE_Y"], landmarks["NOSE_X"])
        lm["topleft"] = (landmarks["FACE_Y"], landmarks["FACE_X"])
        lm["size"] = (landmarks["FACE_X"], landmarks["FACE_W"])

        return lm

    def probes(self):

        if self._cached_probes is None:
            for_models = os.path.join(self.protocol_directory,self.protocol,"dev","for_models.csv")
            for_probes = os.path.join(self.protocol_directory,self.protocol,"dev","for_probes.csv")

            for_models = pd.read_csv(for_models)
            for_probes = pd.read_csv(for_probes)

            if self.same_demo_test:
                grouped_reference_list = for_models.groupby(self.same_demo_test.upper())["PATH"].apply(list).to_dict()
            else:
                reference_list = for_models["PATH"].tolist()

            self._cached_probes = []

            for _, row in for_probes.iterrows():
                row_dict = row.to_dict()
                sset = self._make_sample(row_dict)
                if self.same_demo_test:
                    sset['references'] = grouped_reference_list[sset[self.same_demo_test.lower()]]
                else:
                    sset['references'] = reference_list

                self._cached_probes.append(sset)

        return self._cached_probes

    def references(self):
        if self._cached_biometric_references is None:
            for_models = os.path.join(self.protocol_directory,self.protocol,"dev","for_models.csv")
            for_models = pd.read_csv(for_models)
            self._cached_biometric_references = []
            for _, row in for_models.iterrows():
                row_dict = row.to_dict()
                self._cached_biometric_references.append(self._make_sample(row_dict))

        return self._cached_biometric_references

    def _make_sample(self, item):

        target, subject_id, reference_id = item["PATH"].split("/")
        annotations = self.annotation(item)
        key = item["PATH"]

        path = (
            os.path.join(
                self.data_directory,
                key + self.extension,
            )
        )


        sample = {"path":path,"key":key,"annotation":annotations,"reference_id":reference_id,"subject_id":subject_id,"race":item["RACE"],"gender":item["GENDER"],"annotation":annotations}

        return sample


    def treferences(self):

        if self._cached_treferences is None:
            for_tnorm = os.path.join(self.protocol_directory,self.protocol,"norm","for_models_norm.csv")
            for_tnorm = pd.read_csv(for_tnorm)
            self._cached_treferences = []
            for _, row in for_tnorm.iterrows():
                row_dict = row.to_dict()
                self._cached_treferences.append(self._make_sample(row_dict))

        return self._cached_treferences


    def zprobes(self):
        if self._cached_zprobes is None:

            for_probes = os.path.join(self.protocol_directory,self.protocol,"norm","for_probes_norm.csv")
            for_probes = pd.read_csv(for_probes)

            for_models = os.path.join(self.protocol_directory,self.protocol,"dev","for_models.csv")
            for_models = pd.read_csv(for_models)
            if self.same_demo_train:
                grouped_reference_list = for_models.groupby(self.same_demo_train.upper())["PATH"].apply(list).to_dict()
                # grouped_reference_list = for_models.groupby(["RACE","GENDER"])["PATH"].apply(list).to_dict()
            else:
                reference_list = for_models["PATH"].tolist()
            self._cached_zprobes = []

            for _, row in for_probes.iterrows():
                row_dict = row.to_dict()
                sset = self._make_sample(row_dict)
                if self.same_demo_train:
                    sset['references'] = grouped_reference_list[sset[self.same_demo_train.lower()]]
                    # sset['references'] = grouped_reference_list[(sset["race"],sset["gender"])]
                else:
                    sset['references'] = reference_list

                self._cached_zprobes.append(sset)

        return self._cached_zprobes

    def cohort_based_trainsamples(self):
        if self._cached_coVSco_zprobes is None:
            self._cached_coVSco_zprobes = []
            for_probes = os.path.join(self.protocol_directory,self.protocol,"norm","for_probes_norm.csv")
            for_probes = pd.read_csv(for_probes)
            for_models = os.path.join(self.protocol_directory,self.protocol,"norm","for_models_norm.csv")

            for_models = pd.read_csv(for_models)
            if self.same_demo_train:
                grouped_reference_list = for_models.groupby(self.same_demo_train.upper())["PATH"].apply(list).to_dict()
            else:
                reference_list = for_models["PATH"].tolist()

            for _, row in for_probes.iterrows():
                row_dict = row.to_dict()
                sset = self._make_sample(row_dict)
                if self.same_demo_train:
                    sset['references'] = grouped_reference_list[sset[self.same_demo_train.lower()]]
                else:
                    sset['references'] = reference_list

                self._cached_coVSco_zprobes.append(sset)

        if self._cached_coVSco_references is None:
            self._cached_coVSco_references = self.treferences()

        return self._cached_coVSco_zprobes, self._cached_treferences

    def protocols():
        return [
            "short-demo",
            "full-demo",
        ]

import json
import random
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == "-lrb-"):
        return "("
    elif (token.lower() == "-rrb-"):
        return ")"
    elif (token.lower() == "-lsb-"):
        return "["
    elif (token.lower() == "-rsb-"):
        return "]"
    elif (token.lower() == "-lcb-"):
        return "{"
    elif (token.lower() == "-rcb-"):
        return "}"
    return token


class TACREDDataset(Dataset):
    def __init__(self, data_file, no_task_desc=False):
        self.data = []
        # raw_labelset = ["no_relation", "per:title", "org:top_members/employees", "per:employee_of",
        #                 "org:alternate_names", "org:country_of_headquarters", "per:countries_of_residence",
        #                 "org:city_of_headquarters", "per:cities_of_residence", "per:age",
        #                 "per:stateorprovinces_of_residence", "per:origin", "org:subsidiaries", "org:parents",
        #                 "per:spouse", "org:stateorprovince_of_headquarters", "per:children", "per:other_family",
        #                 "per:alternate_names", "org:members", "per:siblings", "per:schools_attended", "per:parents",
        #                 "per:date_of_death", "org:member_of", "org:founded_by", "org:website", "per:cause_of_death",
        #                 "org:political/religious_affiliation", "org:founded", "per:city_of_death", "org:shareholders",
        #                 "org:number_of_employees/members", "per:date_of_birth", "per:city_of_birth", "per:charges",
        #                 "per:stateorprovince_of_death", "per:religion", "per:stateorprovince_of_birth",
        #                 "per:country_of_birth", "org:dissolved", "per:country_of_death"]
        raw_labelset = ["org:members", "per:siblings", "per:spouse", "org:country_of_branch", "per:country_of_death",
                        "per:parents", "per:stateorprovinces_of_residence", "org:top_members/employees",
                        "org:dissolved", "org:number_of_employees/members", "per:stateorprovince_of_death",
                        "per:origin", "per:children", "org:political/religious_affiliation", "per:city_of_birth",
                        "per:title", "org:shareholders", "per:employee_of", "org:member_of", "org:founded_by",
                        "per:countries_of_residence", "per:other_family", "per:religion", "per:identity",
                        "per:date_of_birth", "org:city_of_branch", "org:alternate_names", "org:website",
                        "per:cause_of_death", "org:stateorprovince_of_branch", "per:schools_attended",
                        "per:country_of_birth", "per:date_of_death", "per:city_of_death", "org:founded",
                        "per:cities_of_residence", "per:age", "per:charges", "per:stateorprovince_of_birth",
                        "no_relation"]
        self.labelset = [self.preprocess_label(label) for label in raw_labelset]

        with open(data_file, "r") as f:
            data = json.load(f)

        for d in tqdm(data, desc="Preprocessing"):
            ss, se, st = d["subj_start"], d["subj_end"] + 1, d["subj_type"].lower()
            os, oe, ot = d["obj_start"], d["obj_end"] + 1, d["obj_type"].lower()

            tokens = [convert_token(token) for token in d["token"]]
            label = d["relation"]
            label = self.preprocess_label(label)
            assert label in self.labelset

            # add marker and type (and task description if needed)
            if no_task_desc:
                if ss < os:
                    sent = tokens[:ss] + ["<SUBJ>"] + [st] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:os] + ["<OBJ>"] + [
                        ot] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:]
                else:
                    sent = tokens[:os] + ["<OBJ>"] + [ot] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:ss] + ["<SUBJ>"] + [
                        st] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:]
            else:
                if ss < os:
                    sent = tokens[:ss] + ["<SUBJ>"] + tokens[ss:se] + ["</SUBJ>"] + tokens[se:os] + ["<OBJ>"] + tokens[
                                                                                                                os:oe] + [
                               "</OBJ>"] + tokens[oe:]
                else:
                    sent = tokens[:os] + ["<OBJ>"] + tokens[os:oe] + ["</OBJ>"] + tokens[oe:ss] + ["<SUBJ>"] + tokens[
                                                                                                               ss:se] + [
                               "</SUBJ>"] + tokens[se:]
                sent += ["</s>", "</s>", "Describe", "the", "relationship", "between"] + [st] + tokens[ss:se] + [
                    "and"] + [ot] + tokens[os:oe] + ["."]

            self.data.append([sent, label])

    def preprocess_label(self, label):
        rep_rule = (
        ("_", " "), ("per:", "person "), ("org:", "organization "), ("stateorprovince", "state or province"))
        for r in rep_rule:
            label = label.replace(*r)
        return label

    def __getitem__(self, idx):
        pos = self.data[idx][1]
        neg = pos
        while neg == pos:
            neg = random.choice(self.labelset)
        return self.data[idx] + [neg]

    def __len__(self):
        return len(self.data)



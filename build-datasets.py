import json
import os
from typing import Literal 
import pandas as pd


def count_func(x: str) -> str:
    x = int(x)
    if x == 0:
        return "0"
    elif 1 <= x <= 10:
        return "between 1 and 10"
    elif 11 <= x <= 100:
        return "between 11 and 100"
    elif 101 <= x <= 1000:
        return "between 101 and 1000"
    elif 1000 < x:
        return "more than 1000"

def area_func(x):
    x =  int(x.split("m2")[0])
    if x == 0:
        return "0 m²"
    elif 1 <= x <= 10:
        return "between 1 m² and 10 m²"
    elif 11 <= x <= 100:
        return "between 11 m² and 100 m²"
    elif 101 <= x <= 1000:
        return "between 101 m² and 1000 m²"
    elif 1000 < x:
        return "more than 1000 m²"

def process_split(questions: list, answers: list, dataset: Literal["lr", "hr"]) -> pd.DataFrame:
    records = {
        "type": [],
        "question": [],
        "img_id": [],
        "img": [],
        "answer": [],
    }

    for question, answer in zip(questions, answers):
        if question["active"] and answer["active"]:
            if question["type"] == "count" and dataset == "lr":
                records["answer"].append(count_func(answer["answer"]))
            elif question["type"] == "area" and dataset == "hr":
                records["answer"].append(area_func(answer["answer"]))
            else:
                records["answer"].append(answer["answer"])

            records["type"].append(question["type"])
            records["question"].append(question["question"])
            records["img_id"].append(question["img_id"])
            records["img"].append(os.path.join("data", f"rsvqa-{dataset}", "images", f"{question['img_id']}.tif"))
    
    return pd.DataFrame.from_records(records)

# rsvqa-lr
# train split
with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_train_answers.json"), "r") as f:
    answers = json.load(f)["answers"]
with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_train_questions.json"), "r") as f:
    questions = json.load(f)["questions"]
process_split(questions, answers, "lr").to_csv(os.path.join("data", "rsvqa-lr", "train.csv"))

# validation split
with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_val_answers.json"), "r") as f:
    answers = json.load(f)["answers"]
with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_val_questions.json"), "r") as f:
    questions = json.load(f)["questions"]
process_split(questions, answers, "lr").to_csv(os.path.join("data", "rsvqa-lr", "validation.csv"))

# test split
with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_test_answers.json"), "r") as f:
    answers = json.load(f)["answers"]
with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_test_questions.json"), "r") as f:
    questions = json.load(f)["questions"]
process_split(questions, answers, "lr").to_csv(os.path.join("data", "rsvqa-lr", "test.csv"))

# rsvqa-hr
# train split
with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_train_answers.json"), "r") as f:
    answers = json.load(f)["answers"]
with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_train_questions.json"), "r") as f:
    questions = json.load(f)["questions"]
process_split(questions, answers, "hr").to_csv(os.path.join("data", "rsvqa-hr", "train.csv"))

# validation split
with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_val_answers.json"), "r") as f:
    answers = json.load(f)["answers"]
with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_val_questions.json"), "r") as f:
    questions = json.load(f)["questions"]
process_split(questions, answers, "hr").to_csv(os.path.join("data", "rsvqa-hr", "validation.csv"))

# test split
with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_test_answers.json"), "r") as f:
    answers = json.load(f)["answers"]
with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_test_questions.json"), "r") as f:
    questions = json.load(f)["questions"]
process_split(questions, answers, "hr").to_csv(os.path.join("data", "rsvqa-hr", "test.csv"))

# test phili split
with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_test_phili_answers.json"), "r") as f:
    answers = json.load(f)["answers"]
with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_test_phili_questions.json"), "r") as f:
    questions = json.load(f)["questions"]
process_split(questions, answers, "hr").to_csv(os.path.join("data", "rsvqa-hr", "test_phili.csv"))

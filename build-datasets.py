import json
import os
from typing import Literal 
from datasets import Dataset, DatasetDict, Image

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

def process_split(questions: list, answers: list, split: str, dataset: Literal["lr", "hr"]) -> Dataset:
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
    
    return Dataset.from_dict(records, split=split)

if __name__=="__main__": 
    # rsvqa-lr
    rsvqa_lr = DatasetDict()
    # train split
    with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_train_answers.json"), "r") as f:
        answers = json.load(f)["answers"]
    with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_train_questions.json"), "r") as f:
        questions = json.load(f)["questions"]
    rsvqa_lr["train"] = process_split(questions, answers, "train", "lr")

    # validation split
    with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_val_answers.json"), "r") as f:
        answers = json.load(f)["answers"]
    with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_val_questions.json"), "r") as f:
        questions = json.load(f)["questions"]
    rsvqa_lr["validation"] = process_split(questions, answers, "validation", "lr")

    # test split
    with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_test_answers.json"), "r") as f:
        answers = json.load(f)["answers"]
    with open(os.path.join("data", "rsvqa-lr", "raw", "LR_split_test_questions.json"), "r") as f:
        questions = json.load(f)["questions"]
    rsvqa_lr["test"] = process_split(questions, answers, "test", "lr")
    
    rsvqa_lr.save_to_disk(os.path.join("data", "rsvqa-hr", "hf"))


    # rsvqa-hr
    rsvqa_hr = DatasetDict()
    # train split
    with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_train_answers.json"), "r") as f:
        answers = json.load(f)["answers"]
    with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_train_questions.json"), "r") as f:
        questions = json.load(f)["questions"]
    rsvqa_hr["train"] = process_split(questions, answers, "train", "hr")

    # validation split
    with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_val_answers.json"), "r") as f:
        answers = json.load(f)["answers"]
    with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_val_questions.json"), "r") as f:
        questions = json.load(f)["questions"]
    rsvqa_hr["validation"] = process_split(questions, answers, "validation", "hr")

    # test split
    with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_test_answers.json"), "r") as f:
        answers = json.load(f)["answers"]
    with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_test_questions.json"), "r") as f:
        questions = json.load(f)["questions"]
    rsvqa_hr["test"] = process_split(questions, answers, "test", "hr")

    # test phili split
    with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_test_phili_answers.json"), "r") as f:
        answers = json.load(f)["answers"]
    with open(os.path.join("data", "rsvqa-hr", "raw", "USGS_split_test_phili_questions.json"), "r") as f:
        questions = json.load(f)["questions"]
    rsvqa_hr["test_phili"] = process_split(questions, answers, "test_phili", "hr")
    
    rsvqa_hr.save_to_disk(os.path.join("data", "rsvqa-hr", "hf"))
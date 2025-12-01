import json
import argparse
from collections import defaultdict
from statistics import mean


def get_best_learning_rate(results_dict):
    best_lr = None
    best_f1 = float("-inf")
    for lr, res in results_dict.items():
        total = res.get("total", {})
        f1_scores = []
        for k,v in total.items():
            if "f1" in k and not "_se" in k:
                f1_scores.append(v)
            if len(f1_scores)>1:
                f1_scores = [mean(f1_scores)]
        f1 = f1_scores[0]
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_lr = lr
    return best_lr, best_f1

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_file',default="euroeval_benchmark_results_hyperparameter_search.jsonl",help="results file")
    ap.add_argument('--output_file',default="best_lrs.jsonl")
    args = ap.parse_args()
    all_results = []
    with open(args.results_file,"r") as f:
        for l in f:
            try:
                all_results.append(json.loads(l))
            except json.JSONDecodeError as e:
                print(e)

    hp_search_results = []
    for entry in all_results:
        res_dict = {}
        results = entry["results"]
        model = entry["model"]
        dataset = entry["dataset"]
        language = entry['dataset_languages'][0]

        best_lr, best_score = get_best_learning_rate(results)
        res_dict['dataset']=dataset
        res_dict['model']=model
        res_dict['language']=language
        res_dict['best_lr']=best_lr
        res_dict['mean_f1']=best_score
        hp_search_results.append(res_dict)


        print(f"Dataset: {dataset}")
        print(f"Language: {language}")
        print(f"Model: {model}")
        print(f"Best learning rate: {best_lr} with mean F1 = {best_score:.2f}")
        print()
    with open(args.output_file,"w") as f:
        for l in hp_search_results:
            f.write(json.dumps(l,ensure_ascii=False) +"\n")
import json
import os
results = []
for f in os.listdir("./"):
    if os.path.isdir(f):
        if "TurkuNLP" in f:
            with open(f"{f}/euroeval_benchmark_results_spesific_lrs.jsonl","r") as r_f:
                for l in r_f:
                    try:
                        r = json.loads(l)
                        results.append(r)
                    except json.decoder.JSONDecodeError as e:
                        print(e)
                
                
with open("partial_res.jsonl","w") as out_f:
    for l in results:
        out_f.write(json.dumps(l,ensure_ascii=False))
        out_f.write("\n")
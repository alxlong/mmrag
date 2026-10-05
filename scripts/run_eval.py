from mmrag.judge import score_retrieval, judge_answer
from mmrag.retrieve import retrieve
from mmrag.generate import generate
from mmrag.images import load_image
import yaml, lancedb, json
from datetime import datetime

with open('evalset/questions.yaml') as f:
    questions = yaml.safe_load(f)

db = lancedb.connect('data/lancedb')
table = db.open_table('CLIP_images')

for question in questions:
    hits = retrieve(table, question['query'], 5)
    images = [load_image(hit["key"], hit["shard_number"]) for hit in hits]

    scores = score_retrieval([hit['key'] for hit in hits], question['relevant_keys'])
    answer = generate(images, question["question"])

    # handles the model returning incorrectly formatted text
    attempts, max_attempts = 0, 5
    while attempts < max_attempts:
        try:
            verdict =  json.loads(judge_answer(question["question"], answer, images))
            break
        except json.decoder.JSONDecodeError:
            attempts += 1
            if attempts < max_attempts:
                print('Invalid JSON; Trying again.\n')
            else:
                raise

    record = {
        "id": question["id"],
        "query": question["query"],
        "question": question["question"],
        "question_type": question["type"],
        "retrieved_keys": [h["key"] for h in hits],
        "retrieved_shards": [int(h["shard_number"]) for h in hits],
        "relevant_keys": question["relevant_keys"],
        "hits": scores["hits"],
        "precision_at_5": scores["precision_at_k"],
        "recall": scores["recall"],
        "rr": scores["rr"],
        "answer": answer,
        "verdict": verdict.get("verdict", []),
        "claims": verdict.get("claims", []),
        "gen_model": "claude-sonnet-4-6",
        "judge_model": "claude-sonnet-4-6",
    }

    with open(f'data/outputs/run_{datetime.now():%Y%m%d}.jsonl', 'a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
import json, os

def save_metrics(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f: json.dump(d, f, indent=2)

def append_run(path, tag, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if os.path.exists(path):
        with open(path) as f: data = json.load(f)
    data[tag] = metrics
    with open(path, 'w') as f: json.dump(data, f, indent=2)

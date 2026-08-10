import json

data = json.load(open("/tmp/cov.json"))
ctxs = set()
for f in data.get("files", {}).values():
    for c in f.get("contexts", {}).values():
        ctxs.update(c)

print(list(ctxs)[:10])

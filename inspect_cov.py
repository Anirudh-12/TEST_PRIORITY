import json

# Try to find a coverage file
import glob
files = glob.glob('/tmp/coverage_*.json')
if not files:
    print("No coverage files found")
else:
    for file in files:
        data = json.load(open(file))
        ctxs = set()
        for f in data.get('files', {}).values():
            for c in f.get('contexts', {}).values():
                ctxs.update(c)
        print(f"File {file} has {len(ctxs)} unique contexts.")
        print(list(ctxs)[:10])

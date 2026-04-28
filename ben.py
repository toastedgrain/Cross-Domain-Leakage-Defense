import json
d = json.load(open('outputs/persistbench/baseline/persist_full_gemini3_pro.json',
                     encoding='utf-8'))
m = 'google/gemini-3.1-pro-preview'
with open('gemini_beneficial_audit.txt', 'w', encoding='utf-8') as out:
    for eid, e in d['entries'].items():
        if e.get('failure_type') != 'beneficial_memory_usage':
            continue
        g = e['results'][m]['generations'][0]
        out.write(f"=== {eid[:8]} | score={g['judge']['score']} ===\n")
        out.write("Q: " + e['query'] + "\n")
        out.write("MEMORIES:\n")
        for mem in e['memories']:
            out.write(f"  - {mem}\n")
        out.write("RESPONSE:\n" + g['memory_response'] + "\n")
        out.write("JUDGE: " + g['judge']['reasoning'] + "\n\n")
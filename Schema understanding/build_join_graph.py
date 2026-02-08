import json
from collections import defaultdict

with open("schema_catalog.json", encoding="utf-8") as f:
    schema = json.load(f)

graph = defaultdict(list)

for table, meta in schema["tables"].items():
    for fk in meta["fks"]:
        graph[table].append(fk["ref_table"])

join_graph = {
    "domain": "curriculum",
    "edges": dict(graph)
}

with open("join_graph.json", "w", encoding="utf-8") as f:
    json.dump(join_graph, f, indent=2)

print("✅ join_graph.json generated")

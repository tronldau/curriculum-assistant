import csv, json
from collections import defaultdict

tables = defaultdict(lambda: {
    "columns": [],
    "pk": [],
    "fks": []
})

# 1. load columns
with open("columns.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        tables[r["TABLE_NAME"]]["columns"].append(r["COLUMN_NAME"])

# 2. load primary keys
with open("pk.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
       tables[r["TABLE_NAME"]]["pk"].append(r["COLUMN_NAME"])

# 3. load foreign keys
with open("fk.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        tables[r["TABLE_NAME"]]["fks"].append({
    "column": r["COLUMN_NAME"],
    "ref_table": r["REFERENCED_TABLE_NAME"],
    "ref_column": r["REFERENCED_COLUMN_NAME"]
})


schema_catalog = {
    "domain": "curriculum",
    "tables": tables
}

with open("schema_catalog.json", "w", encoding="utf-8") as f:
    json.dump(schema_catalog, f, indent=2)

print("✅ schema_catalog.json generated")

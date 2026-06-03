"""导出 Neo4j 全部 Recipe.name 到 testset_output/_kb_recipe_names.txt（OOD 避让用，不入仓库）"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neo4j import GraphDatabase
from config import DEFAULT_CONFIG as C

drv = GraphDatabase.driver(C.neo4j_uri, auth=(C.neo4j_user, C.neo4j_password))
names = []
with drv.session(database=C.neo4j_database) as s:
    for r in s.run("MATCH (r:Recipe) WHERE r.name IS NOT NULL RETURN DISTINCT r.name AS n"):
        names.append(r["n"])
drv.close()
out = os.path.join(os.path.dirname(__file__), "..", "testset_output", "_kb_recipe_names.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(names)))
print(f"导出 {len(names)} 个菜名 → {out}")

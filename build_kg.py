"""Minimal KG builder template for Assignment 4.

Keep this contract unchanged:
- Graph: (Regulation)-[:HAS_ARTICLE]->(Article)-[:CONTAINS_RULE]->(Rule)
- Article: number, content, reg_name, category
- Rule: rule_id, type, action, result, art_ref, reg_name
- Fulltext indexes: article_content_idx, rule_idx
- SQLite file: ncu_regulations.db
"""

import os
import sqlite3
import json
import re
import uuid
from typing import Any
from dotenv import load_dotenv
from neo4j import GraphDatabase
from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline

load_dotenv()
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

def extract_entities(article_number: str, reg_name: str, content: str) -> dict[str, Any]:
    """สกัดเงื่อนไขทางกฎหมายโดยใช้ Local LLM"""
    tok = get_tokenizer()
    pipe = get_raw_pipeline()
    if tok is None or pipe is None:
        load_local_llm()
        tok = get_tokenizer()
        pipe = get_raw_pipeline()

    sys_prompt = """You are a legal data extractor. Extract the rules from the given article.
Output ONLY a valid JSON object in this format:
{
  "rules": [
    {
      "type": "obligation/permission/prohibition",
      "action": "what action is described",
      "result": "the condition or consequence"
    }
  ]
}
Do not add any explanations or markdown."""
    
    user_prompt = f"Regulation: {reg_name}\nArticle: {article_number}\nContent: {content}"
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = pipe(prompt, max_new_tokens=256)[0]["generated_text"].strip()

    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(response)
    except:
        # Fallback หาก LLM สร้าง JSON พัง
        return {"rules": [{"type": "general", "action": content[:100], "result": "Please refer to the full article."}]}

def build_fallback_rules(article_number: str, content: str) -> list[dict[str, str]]:
    return []

def build_graph() -> None:
    sql_conn = sqlite3.connect("ncu_regulations.db")
    cursor = sql_conn.cursor()
    driver = GraphDatabase.driver(URI, auth=AUTH)

    print("Loading Local LLM...")
    load_local_llm()

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        # 1) Regulation nodes
        cursor.execute("SELECT reg_id, name, category FROM regulations")
        regulations = cursor.fetchall()
        reg_map = {}
        for reg_id, name, category in regulations:
            reg_map[reg_id] = (name, category)
            session.run(
                "MERGE (r:Regulation {id:$rid}) SET r.name=$name, r.category=$cat",
                rid=reg_id, name=name, cat=category
            )

        # 2) Article nodes
        cursor.execute("SELECT reg_id, article_number, content FROM articles")
        articles = cursor.fetchall()
        for reg_id, article_number, content in articles:
            reg_name, reg_category = reg_map.get(reg_id, ("Unknown", "Unknown"))
            session.run(
                """
                MATCH (r:Regulation {id: $rid})
                CREATE (a:Article {number: $num, content: $content, reg_name: $reg_name, category: $reg_category})
                MERGE (r)-[:HAS_ARTICLE]->(a)
                """,
                rid=reg_id, num=article_number, content=content, reg_name=reg_name, reg_category=reg_category
            )

        session.run("CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS FOR (a:Article) ON EACH [a.content]")

        print(f"Extracting rules for {len(articles)} articles (This may take a while)...")
        rule_counter = 0

        # TODO(student): Iterate and create rules
        for reg_id, article_number, content in articles:
            reg_name, _ = reg_map.get(reg_id, ("Unknown", "Unknown"))
            extracted = extract_entities(article_number, reg_name, content)
            
            for rule in extracted.get("rules", []):
                action = rule.get("action", "")
                result = rule.get("result", "")
                r_type = rule.get("type", "general")
                
                if not action and not result:
                    continue
                
                rule_id = str(uuid.uuid4())
                session.run(
                    """
                    MATCH (a:Article {number: $num, reg_name: $reg_name})
                    CREATE (r:Rule {
                        rule_id: $rule_id, type: $type, action: $action, 
                        result: $result, art_ref: $num, reg_name: $reg_name
                    })
                    MERGE (a)-[:CONTAINS_RULE]->(r)
                    """,
                    num=article_number, reg_name=reg_name, rule_id=rule_id, 
                    type=r_type, action=action, result=result
                )
                rule_counter += 1

        session.run("CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS FOR (r:Rule) ON EACH [r.action, r.result]")

        coverage = session.run("""
            MATCH (a:Article) OPTIONAL MATCH (a)-[:CONTAINS_RULE]->(r:Rule)
            WITH a, count(r) AS rule_count
            RETURN count(a) AS total_articles,
                   sum(CASE WHEN rule_count > 0 THEN 1 ELSE 0 END) AS covered_articles,
                   sum(CASE WHEN rule_count = 0 THEN 1 ELSE 0 END) AS uncovered_articles
        """).single()
        print(f"[Coverage] covered={coverage['covered_articles']}/{coverage['total_articles']}, uncovered={coverage['uncovered_articles']}")

    driver.close()
    sql_conn.close()

if __name__ == "__main__":
    build_graph()
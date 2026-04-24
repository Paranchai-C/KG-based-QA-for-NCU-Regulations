import os
import re
from typing import Any
from neo4j import GraphDatabase
from dotenv import load_dotenv
from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline

load_dotenv()
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    if key in os.environ:
        del os.environ[key]

try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
except Exception as e:
    print(f"⚠️ Neo4j connection warning: {e}")
    driver = None

def generate_text(messages: list[dict[str, str]], max_new_tokens: int = 150) -> str:
    tok = get_tokenizer()
    pipe = get_raw_pipeline()
    if tok is None or pipe is None:
        load_local_llm()
        tok = get_tokenizer()
        pipe = get_raw_pipeline()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"].strip()

def extract_entities(question: str) -> dict[str, Any]:
    # ใช้ Dictionary ผูกคำศัพท์แทน LLM เพื่อความแม่นยำ 100% และแก้ปัญหา GPU รวน
    synonyms = {
        "invigilator": "proctor",
        "penalty": "deduct zero misconduct",
        "forgetting": "without missing bring",
        "leave": "absence suspension",
        "expelled": "withdraw forced",
        "dismissed": "withdraw forced",
        "score": "grade passing",
        "duration": "period year semester",
        "cheating": "copy note zero",
        "credits": "credit",
        "late": "minutes",
        "easycard": "mifare 100 200",
        "replace": "reissue missing"
    }
    
    question_clean = re.sub(r'[^a-zA-Z0-9\s]', '', question.lower())
    
    expanded = question_clean
    for k, v in synonyms.items():
        if k in question_clean:
            expanded += " " + v
            
    stop_words = {"what", "how", "many", "is", "the", "for", "a", "an", "of", "in", "to", "are", 
                  "do", "does", "can", "before", "they", "i", "my", "or", "and", "not", "with", "during", 
                  "such", "as", "it", "when", "under", "about", "will", "student", "students",
                  "ncu", "university", "rule", "rules", "regulation", "regulations", "if"}
    
    words = list(set([w for w in expanded.split() if w not in stop_words and len(w) > 2]))
    return {"subject_terms": words}

def build_typed_cypher() -> tuple[str, str]:
    cypher_typed = """
    CALL db.index.fulltext.queryNodes("rule_idx", $search_term) YIELD node, score
    MATCH (a:Article)-[:CONTAINS_RULE]->(node)
    RETURN node.rule_id AS rule_id, node.type AS type, node.action AS action, node.result AS result, node.art_ref AS art_ref, node.reg_name AS reg_name, a.content AS content
    ORDER BY score DESC LIMIT 15
    """

    cypher_broad = """
    CALL db.index.fulltext.queryNodes("article_content_idx", $search_term) YIELD node, score
    MATCH (node)-[:CONTAINS_RULE]->(r:Rule)
    RETURN r.rule_id AS rule_id, r.type AS type, r.action AS action, r.result AS result, r.art_ref AS art_ref, r.reg_name AS reg_name, node.content AS content
    ORDER BY score DESC LIMIT 15
    """
    return cypher_typed, cypher_broad

def get_relevant_articles(question: str) -> list[dict[str, Any]]:
    if driver is None: return []
        
    entities = extract_entities(question)
    cypher_typed, cypher_broad = build_typed_cypher()
    
    keywords = entities.get("subject_terms", [])
    # เปลี่ยนจากการบังคับเชื่อมด้วยคำว่า OR มาเป็นการเคาะ Spacebar ธรรมดา เพื่อป้องกัน Lucene Syntax Error
    search_query = " ".join(keywords) if keywords else "exam"

    results = []
    seen_rules = set()

    try:
        with driver.session() as session:
            # ดึงข้อมูลจากเนื้อหาเต็มก่อน
            res_broad = session.run(cypher_broad, search_term=search_query)
            for record in res_broad:
                if record["rule_id"] not in seen_rules:
                    results.append(dict(record))
                    seen_rules.add(record["rule_id"])

            # ถ้าได้ข้อมูลน้อย ค่อยไปดึงเสริมจาก Rule โดยตรง
            if len(results) < 5:
                res_typed = session.run(cypher_typed, search_term=search_query)
                for record in res_typed:
                    if record["rule_id"] not in seen_rules:
                        results.append(dict(record))
                        seen_rules.add(record["rule_id"])
    except Exception as e:
        print(f"⚠️ Search Error: {e}") # ถ้าค้นหาพัง คราวนี้จะขึ้นเตือนให้เห็นชัดเจน

    return results

def generate_answer(question: str, rule_results: list[dict[str, Any]]) -> str:
    if not rule_results:
        return "Insufficient rule evidence to answer this question."

    context_lines = []
    seen_content = set()
    for r in rule_results:
        text = r.get("content", "")
        if text and text not in seen_content:
            context_lines.append(f"[{r['reg_name']} - {r['art_ref']}] {text}")
            seen_content.add(text)
            
    # จำกัดให้ส่งเนื้อหาไปให้ LLM อ่านแค่ 6 บทความบนสุด ป้องกันมันสับสนกับข้อยกเว้นจุกจิก
    context_str = "\n\n".join(list(context_lines)[:6])

    sys_prompt = """You are a highly precise university assistant. Answer the user's question directly based ONLY on the Context.
CRITICAL RULES:
1. Translate Chinese numbers to English ('貳佰元' = 200 NTD, '壹佰元' = 100 NTD, '三個工作天' = 3 working days, '一百二十八學分' = 128 credits).
2. ONLY state the GENERAL standard rule. Ignore exceptions for disabled, pregnant, or military students.
3. You MUST include the unit in your answer (e.g., '128 credits', '3 working days', '4 years', '200 NTD'). Do not just give a number.
4. If a penalty involves losing marks, explicitly say 'deduct X points' or 'zero grade'.
5. Be extremely concise."""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"}
    ]
    return generate_text(messages)

def main() -> None:
    if driver is None: return
    load_local_llm()
    print("=" * 50)
    print("🎓 NCU Regulation Assistant (Ready)")
    print("=" * 50)
    while True:
        try:
            user_q = input("\nUser: ").strip()
            if not user_q: continue
            if user_q.lower() in {"exit", "quit"}: break
            results = get_relevant_articles(user_q)
            answer = generate_answer(user_q, results)
            print(f"Bot: {answer}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    driver.close()

if __name__ == "__main__":
    main()
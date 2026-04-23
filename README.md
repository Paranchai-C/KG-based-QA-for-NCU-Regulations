# Assignment 4: KG-based QA for NCU Regulations

**Author:** Paranchai Chianvichai (潘志凱) 114522606  
**Department:** Computer Science and Information Engineering (CSIE), National Central University

---

## 1. Project Overview
This project implements a Knowledge Graph-based Question Answering (QA) system for NCU university regulations. The system extracts legal rules from raw PDF documents, stores them in a Neo4j Knowledge Graph, and utilizes a local Large Language Model (HuggingFace `Qwen/Qwen2.5-3B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`) to retrieve context and generate accurate, grounded answers.

## 2. KG Construction Logic and Schema Design
The Knowledge Graph is built using a rigid hierarchical schema to ensure that rules are always traceable back to their source articles and regulations.

### Schema Design
The graph follows this exact structure:
`(:Regulation)-[:HAS_ARTICLE]->(:Article)-[:CONTAINS_RULE]->(:Rule)`

* **Regulation Node:** Represents the overarching legal document (e.g., "NCU Student Examination Rules").
    * Properties: `id`, `name`, `category`
* **Article Node:** Represents a specific section or article within the regulation.
    * Properties: `number`, `content`, `reg_name`, `category`
* **Rule Node:** Represents an atomic legal condition extracted by the LLM.
    * Properties: `rule_id`, `type`, `action`, `result`, `art_ref`, `reg_name`

### Extraction Logic
During the KG build phase (`build_kg.py`), the system iterates through each Article in the SQLite database and prompts the local LLM to extract "Rules" in a strictly formatted JSON structure (specifying `type`, `action`, and `result`). A regex fallback mechanism is implemented to handle cases where the LLM outputs malformed JSON.

### KG Visualization
*(Below is the Neo4j Browser screenshot showing Regulation -> Article -> Rule nodes)*

[INSERT YOUR NEO4J GRAPH SCREENSHOT HERE]

---

## 3. Key Cypher Query Design & Retrieval Strategy
To maximize retrieval precision and recall, the system employs a **Two-Stage Hybrid Retrieval Strategy** utilizing Neo4j's Full-Text Search capabilities.

### Stage 1: Typed Rule Retrieval (High Precision)
The system first extracts core keywords from the user's question using the LLM. It then queries the `rule_idx` (which indexes the `action` and `result` properties of Rule nodes).

```cypher
CALL db.index.fulltext.queryNodes("rule_idx", $query) YIELD node, score
RETURN node.rule_id AS rule_id, node.type AS type, node.action AS action, 
       node.result AS result, node.art_ref AS art_ref, node.reg_name AS reg_name
ORDER BY score DESC LIMIT 4
```

### Stage 2: Broad Article Fallback (High Recall)
If the strict rule retrieval yields less than 3 results, the system falls back to querying the `article_content_idx` (which indexes the full text of the Article node). It then traverses the graph to find the associated rules via the `[:CONTAINS_RULE]` relationship.

```cypher
CALL db.index.fulltext.queryNodes("article_content_idx", $query) YIELD node, score
MATCH (node)-[:CONTAINS_RULE]->(r:Rule)
RETURN r.rule_id AS rule_id, r.type AS type, r.action AS action, 
       r.result AS result, r.art_ref AS art_ref, r.reg_name AS reg_name
ORDER BY score DESC LIMIT 4
```

---

## 4. Failure Analysis & Improvements Made

During the development and testing phases, several issues were encountered and resolved:

1. **LLM Output Formatting Failure:** * *Issue:* The local LLM occasionally generated extra conversational text alongside the requested JSON, breaking the parsing step in `build_kg.py`.
   * *Improvement:* Implemented a Python Regular Expression (`re.search(r'\{.*\}', response, re.DOTALL)`) to isolate and extract only the JSON block from the generated text. Also added a fallback rule generation if parsing completely fails.
2. **Low Recall on Vague Questions:** * *Issue:* Directly matching user questions to the `Rule` node's short text often missed relevant context.
   * *Improvement:* Introduced the LLM-based Keyword Extractor (`extract_entities`) to clean the user query into 1-3 core keywords before passing it to the Cypher query. Added the Broad Article Fallback strategy to catch keywords hidden in the main article text.
3. **GPU Memory Bottleneck during KG Build:** * *Issue:* Sequential processing of 159 articles using the 3B model was exceptionally slow on limited VRAM.
   * *Improvement:* Added progress tracking and dynamically managed the `device_map="auto"` configuration to ensure stable GPU utilization. Downgraded to the 1.5B model when necessary for faster extraction without sacrificing structural integrity.

---

## 5. Evaluation Results (auto_test.py)

The system was evaluated using the provided `test_data.json` containing benchmark questions. The LLM-as-a-judge evaluated the grounded answers.

* **Total Questions:** 20
* **Passed:** [INSERT PASSED COUNT]
* **Failed:** [INSERT FAILED COUNT]
* **Overall Accuracy:** [INSERT %]%

---

## 6. How to Run

### Prerequisites
* Python 3.11
* Docker Desktop (for Neo4j)

### Step 1: Start Neo4j
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### Step 2: Environment Setup
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Step 3: Execution Order
```bash
python setup_data.py
python build_kg.py
python auto_test.py
```

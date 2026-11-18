# Multi-Agent Workflow Specification

## 1. Multi-Agent System Overview

The application incorporates four specialized autonomous AI agents cooperating in a deterministic pipeline:

```
[ User Query ]
       |
       v
+------------------+
|   Safety Guard   | ---- (Blocked if injection detected)
+--------+---------+
         | (Safe)
         v
+------------------+
|  Planner Agent   | ---- Intent, subtasks, search queries & document filters
+--------+---------+
         |
         v
+------------------+
| Retriever Agent  | ---- Vector similarity search, deduplication & citations
+--------+---------+
         |
         v
+------------------+
| Reasoning Agent  | ---- Cross-document synthesis & cited response
+--------+---------+
         |
         v
+------------------+
| Validator Agent  | ---- Hallucination audit, confidence score & grounding status
+--------+---------+
         |
         v
[ Verified Final Response with Citations & Telemetry Trace ]
```

---

## 2. Agent Roles and Specifications

### 2.1 Planner Agent (`PlannerAgent`)
- **Primary Objective**: Analyze raw natural language queries to formulate an execution plan.
- **Tasks**:
  1. Classify intent: `Factual Lookup`, `Comparative`, `Tabular / Aggregation`, `Summarization`, `Exploratory`.
  2. Extract key domain entities and produce 1–3 targeted semantic search queries.
  3. Identify if the query is targeted at a specific file (e.g. "according to contract.docx").
  4. Formulate ordered subtasks for the subsequent agents.
- **Output Schema**:
  ```json
  {
    "intent": "Tabular Calculation & Extraction",
    "search_queries": ["Q3 regional sales revenue", "North America regional total"],
    "document_filter": "sales_q3.xlsx",
    "subtasks": [
      "Retrieve North America revenue records",
      "Verify row indices and figures",
      "Calculate total and draft cited response"
    ]
  }
  ```

---

### 2.2 Retriever Agent (`RetrieverAgent`)
- **Primary Objective**: Fetch and rank the most relevant semantic text and tabular chunks.
- **Tasks**:
  1. Executes multi-query vector searches using ChromaDB.
  2. Applies metadata filters (source file, file type).
  3. Deduplicates passages retrieved across multiple queries.
  4. Ranks chunks by cosine similarity score.
  5. Assembles citations with location coordinates:
     - PDF: `[1] contract.pdf (Page 4)`
     - Excel / CSV: `[2] budget.xlsx (Sheet '2024', Row 12)`
     - Word / TXT: `[3] policy.docx (General)`

---

### 2.3 Reasoning Agent (`ReasoningAgent`)
- **Primary Objective**: Contextual inference and comprehensive response formulation.
- **Tasks**:
  1. Ingests the user query, intent, and formatted document context blocks.
  2. Strictly grounds every factual claim in the retrieved excerpts.
  3. Inserts in-line citation tags corresponding to the source references (e.g. `[1]`, `[2]`).
  4. Handles tabular numbers with mathematical precision.
  5. Transparently notes when requested details are absent from the context.

---

### 2.4 Validator Agent (`ValidatorAgent`)
- **Primary Objective**: Factuality verification and hallucination auditing.
- **Tasks**:
  1. Inspects the Reasoning Agent's draft against the source context blocks.
  2. Evaluates whether any claim is unsupported or invented.
  3. Assigns a status:
     - `VERIFIED`: All assertions backed by document citations.
     - `PARTIALLY_VERIFIED`: General answer supported, but minor extrapolations observed.
     - `UNSUPPORTED`: Crucial claims cannot be substantiated from context.
  4. Computes a quantitative Grounding Confidence Score (0.0 to 1.0).
- **Output Schema**:
  ```json
  {
    "status": "VERIFIED",
    "confidence_score": 0.96,
    "factuality_assessment": "All figures match the indexed spreadsheet exactly.",
    "supported_claims": ["Hardware budget is $2.5M", "Personnel budget is $1.2M"],
    "unsupported_claims": []
  }
  ```

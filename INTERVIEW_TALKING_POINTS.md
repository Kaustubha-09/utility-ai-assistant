# utility-ai-assistant — Interview Talking Points

**Why keyword routing instead of an LLM classifier.** Routing decisions happen on every query. An LLM router costs ~50–200ms and a fraction of a cent per request, where a keyword scorer runs in microseconds for free. More importantly: a keyword router is *auditable*. When the assistant routes a query to the wrong place, I can read the keyword score and know exactly why. An LLM router is a black box. The architecture documents the route decision in debug mode so when something feels off, I can verify.

**Why TF-IDF, not embeddings.** The corpus is 7 policy sections — a few hundred tokens total. Embeddings would mean a vector DB, an embedding API call per document at index time, and another per query at retrieval time. TF-IDF runs in scikit-learn in milliseconds with no external dependencies. The retrieval contract is `top-k by cosine similarity`; if the corpus grew past a thousand chunks, swap the implementation to FAISS + sentence-transformers without changing the contract.

**Grounded synthesis as a hallucination defense.** The system prompt is explicit: *"For billing questions, base your answer ONLY on provided `[TOOL DATA]` and `[POLICY DOCS]`. Do not invent numbers, rates, or policies."* The LLM never sees raw customer data; it sees structured JSON that the tools produced. If the tool errored, the LLM gets the error message and degrades gracefully. The architectural separation between "what do we know" and "how do we explain it" is the defense against the classic *"LLM confidently fabricates a peak-pricing rate"* failure mode.

**Explicit confidence labeling.** The model appends `CONFIDENCE: HIGH/MEDIUM/LOW` to every response. A deterministic parser extracts it. The UI renders it as a color-coded badge. The reason this matters: most LLM apps treat the model's response as equally confident regardless of context. We make uncertainty a first-class field — `LOW` confidence means *no useful context retrieved*, which is the user-visible "I don't know" path that lots of demos skip.

**Anthropic → Gemini migration as a one-file change.** `app/llm.py` was originally `anthropic` + Claude. The current commit migrated to `google-generativeai` + Gemini 2.5 Flash. Routing, tools, RAG, response shape, and confidence parsing didn't change — `llm.py` is a protocol-style boundary. The migration was a one-file diff. This is what *boundary-driven architecture* buys you: vendor swaps are tractable.



# sec_extraction

`sec_extraction_langgraph.ipynb` runs structured extraction or question-driven summarization over SEC filings (10-K, S-1) with Azure Content Understanding + Azure OpenAI (`gpt-6-astra`, chat completions) + LangGraph. Put a filing PDF next to the notebook, edit the config cell (section 2), set `AZURE_OPENAI_KEY` and `AZURE_CU_KEY` (or enter them when prompted), and run top to bottom. `python -m pytest sec_extraction/smoke_test.py -v` runs the offline tests (fake LLM, no network). Design: `docs/superpowers/specs/2026-10-30-sec-filing-extraction-notebook-design.md`.

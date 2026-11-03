"""Run logging: append-only JSONL events per run (writer), run headers (header), the start_run() entry point
(session), Ollama metadata (ollama), and the LiteLLM callback (litellm_cb, which imports LiteLLM and so is not
imported here). LLM calls themselves live in hbench.llm."""

from .writer import BlobStore, RunLog, make_run_id, read_events, sha256_json  # noqa: F401

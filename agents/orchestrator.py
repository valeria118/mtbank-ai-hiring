"""Оркестрация 4 агентов."""
from __future__ import annotations

import asyncio
from typing import Any

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.llm_client import LLMClient
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent
from logging_utils import get_logger, timed

logger = get_logger("agents.orchestrator")

class AgentOrchestrator:
    def __init__(self, llm_client: LLMClient | None = None):
        llm_client = llm_client or LLMClient()
        self.agents = {
            "classification": ClassifierAgent(llm_client),
            "quality_score": QualityAgent(llm_client),
            "compliance": ComplianceAgent(llm_client),
            "summary": SummarizerAgent(llm_client),
        }

    async def run_all(self, transcript: list[dict[str, Any]]) -> dict[str, Any]:
        with timed(logger, "orchestrator.run_all", n_segments=len(transcript)):
            keys = list(self.agents.keys())
            results = await asyncio.gather(
                *(self.agents[k].run(transcript) for k in keys),
                return_exceptions=True,
            )

        output: dict[str, Any] = {}
        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                logger.error(
                    "agent.failed",
                    extra={"extra_data": {"agent": key, "error": str(result)}},
                )
                output[key] = {"error": str(result)}
            else:
                output[key] = result

        summary_result = output.pop("summary", {}) or {}
        if summary_result.get("error"):
            output["summary"] = None
            output["action_items"] = []
            output["summary_error"] = summary_result["error"]
        else:
            output["summary"] = summary_result.get("summary", "")
            output["action_items"] = summary_result.get("action_items", [])
        return output

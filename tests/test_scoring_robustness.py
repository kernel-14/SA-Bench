"""Offline tests for the repo-keyed score cache and the judge batch fallback."""

import json
from pathlib import Path
from types import SimpleNamespace

from sa_bench import judge
from sa_bench.cache import ScoreCache, repo_fingerprint
from sa_bench.types import Dimension, EvidenceItem, EvidenceReport, PipelineConfig, SAUClaim, ScoredClaim


def _claim(i: int) -> SAUClaim:
    return SAUClaim(id=f"c{i}", claim=f"Learning rate is 3e-{i}", source="Sec 4", dimension=Dimension.D1)


def _report(i: int) -> EvidenceReport:
    item = EvidenceItem(file="train.py", lines="1-8", snippet=(f"def train(model, data):\n    lr = 3e-{i}\n    opt = SGD(model.parameters(), lr=lr)\n"
                                   "    for batch in data:\n        loss = model(batch).mean()\n        loss.backward()\n"
                                   "        opt.step()\n    return model"),
                        relevance="direct", kind="code")
    return EvidenceReport(claim_id=f"c{i}", status="found", search_rounds=1, evidence=[item], search_trace=[])


def test_cache_is_keyed_by_repo_content(tmp_path: Path):
    repo_a, repo_b = tmp_path / "a", tmp_path / "b"
    for repo, lr in ((repo_a, "1e-3"), (repo_b, "3e-4")):
        repo.mkdir()
        (repo / "train.py").write_text(f"lr = {lr}\n")
    key_a, key_b = repo_fingerprint(repo_a), repo_fingerprint(repo_b)
    assert key_a != key_b
    cache = ScoreCache(tmp_path / "cache")
    cache.set("paper", ScoredClaim(id="c1", claim="x", source="s", score=1.0, search_rounds=1, judge_reasoning="r"), key_a)
    assert cache.get("paper", "c1", key_a).score == 1.0
    assert cache.get("paper", "c1", key_b) is None
    (repo_a / "train.py").write_text("lr = 5e-4\n")
    assert repo_fingerprint(repo_a) != key_a


class _FakeClient:
    """Returns an empty body for multi-claim prompts and a valid score for single-claim prompts."""

    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, messages, **kwargs):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        if "claims" in payload:
            content = ""
        else:
            content = json.dumps({"claim_id": payload["claim_id"], "score": 0.5, "reasoning": "train.py:1",
                                  "evidence_refs": ["train.py:1-8"]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_failed_batch_falls_back_to_single_claims(monkeypatch):
    monkeypatch.setenv("SAU_JUDGE_MAX_ATTEMPTS", "1")
    client = _FakeClient()
    items = [(_claim(i), _report(i)) for i in range(1, 4)]
    results = judge.judge_claim_batch(items, config=PipelineConfig(judge_batch_size=4, judge_workers=1), client=client)
    assert [r.claim_id for r in results] == ["c1", "c2", "c3"]
    assert all(r.score == 0.5 for r in results)
    assert client.calls == 4  # one failed batch call + three single-claim calls

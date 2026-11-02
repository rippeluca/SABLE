"""The graph export. No API calls."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import RDF

from sable.export_graph import (
    ID,
    ROOT,
    SABLE,
    Row,
    build_graph,
    decisions_from_jsonl,
    load_rows,
    write_graph,
)

QUERIES = ROOT / "queries"


def row(**overrides) -> Row:
    base = dict(
        claim_id="c001",
        doi="10.3391/ai.2021.16.1.07",
        statement="Densities were lower at western sites.",
        evidence="densities of both life stages were lower at western sites.",
        gold="supported",
        theme="freshwater",
        title="Spatial variation in propagule pressure",
        predicted="supported",
        evidence_in_topk=True,
        decision="accept",
        accepted_label="supported",
        token_probability=1.0,
        quotes_ok=True,
        label_changed=False,
        label_after_deletion="supported",
    )
    base.update(overrides)
    return Row(**base)


def local_ids(result, variable: str) -> set[str]:
    found = set()
    for record in result:
        parts = str(getattr(record, variable)).rstrip("/").split("/")
        found.add(parts[-2] if parts[-1] == "evidence" else parts[-1])
    return found


class ExportGraphTests(unittest.TestCase):
    def example(self) -> list[Row]:
        return [
            row(),
            row(
                claim_id="c002",
                statement="Flow velocity controls establishment.",
                evidence="Further studies are needed.",
                gold="unclear",
                predicted="unclear",
                decision="abstain",
                accepted_label=None,
                quotes_ok=False,
            ),
            row(
                claim_id="c010",
                doi="10.3390/d12020082",
                statement="Invasive watermilfoil reduced standing crops.",
                evidence="A span that denies the claim.",
                gold="not supported",
                title="Another paper",
                predicted="not supported",
                decision="abstain",
                accepted_label=None,
                label_changed=True,
                label_after_deletion="supported",
            ),
        ]

    def test_example_graph_answers_the_three_queries(self):
        graph = build_graph(self.example())
        parsed = Graph()
        parsed.parse(data=graph.serialize(format="turtle"), format="turtle")

        kept = parsed.query((QUERIES / "supported_and_unclear.rq").read_text(encoding="utf-8"))
        self.assertEqual(local_ids(kept, "hypothesis"), {"c001", "c002"})

        changed = parsed.query((QUERIES / "label_changed.rq").read_text(encoding="utf-8"))
        self.assertEqual(local_ids(changed, "hypothesis"), {"c010"})

        spans = parsed.query((QUERIES / "evidence_from_paper.rq").read_text(encoding="utf-8"))
        self.assertEqual(local_ids(spans, "evidence"), {"c001", "c002"})

    def test_entropy_and_agreement_are_stored_on_the_hypothesis(self):
        graph = build_graph([row(token_entropy=0.25, agreement=0.8)])
        text = graph.serialize(format="turtle")
        self.assertIn("sable:tokenEntropy", text)
        self.assertIn("sable:agreement", text)
        self.assertIn("0.25", text)
        self.assertIn("0.8", text)

    def test_jsonl_passages_are_not_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in [
                {
                    "claim_id": "c001",
                    "tool": "retrieve",
                    "output": {"passages": [{"text": "SENTINEL_PASSAGE_TEXT"}]},
                },
                {
                    "claim_id": "c001",
                    "tool": "decision",
                    "output": {"decision": "accept", "label": "supported", "token_probability": 1.0, "quotes_ok": True},
                },
            ]))
            decisions = decisions_from_jsonl(path)
        self.assertNotIn("SENTINEL_PASSAGE_TEXT", json.dumps(decisions))
        self.assertEqual(decisions["c001"]["label"], "supported")

    def test_repository_graph_matches_the_runs(self):
        rows = load_rows()
        self.assertEqual(len(rows), 67)
        fresh = build_graph(rows)
        text = fresh.serialize(format="turtle")
        self.assertNotIn("Passage 1", text)
        self.assertNotIn("SENTINEL_PASSAGE_TEXT", text)

        stored = Graph()
        stored.parse(ROOT / "graph" / "hypotheses.ttl")
        self.assertTrue(fresh.isomorphic(stored))

        changed = stored.query((QUERIES / "label_changed.rq").read_text(encoding="utf-8"))
        self.assertEqual(local_ids(changed, "hypothesis"), {"c010", "c062"})

        spans = stored.query((QUERIES / "evidence_from_paper.rq").read_text(encoding="utf-8"))
        self.assertEqual(local_ids(spans, "hypothesis"), {"c001", "c002"})

        labels = {
            str(record.predicted)
            for record in stored.query((QUERIES / "supported_and_unclear.rq").read_text(encoding="utf-8"))
        }
        self.assertEqual(labels, {"supported", "unclear"})
        entropies = list(stored.query(
            "PREFIX sable: <https://github.com/Acul01/SABLE/vocab#> "
            "SELECT ?hypothesis WHERE { ?hypothesis sable:tokenEntropy ?entropy ; sable:agreement ?agreement }"
        ))
        self.assertEqual(len(entropies), 67)

    def test_write_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hypotheses.ttl"
            write_graph(path, self.example())
            parsed = Graph()
            parsed.parse(path)
            self.assertIn("densities of both life stages", path.read_text(encoding="utf-8"))
            self.assertIn((ID["c001"], RDF.type, SABLE.Hypothesis), parsed)

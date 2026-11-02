"""Write the runs out as an RDF graph.

The graph records labels that are already in the tables. It does not judge a
claim again, and it does not extract species, factors, or causal relations.
Papers are DOI URIs. Evidence text is the annotator's span, not a passage
from the log. The one-round log is read only for its decision lines.

cito:supports and cito:disputes follow the human label.
sable:predictedLabel is the temperature-0 label.
sable:tokenProbability is the probability of the first label token in the
one-round run. The other token probabilities were not stored, so the graph
has no entropy for that distribution.
sable:labelChanged is present only where the span was deleted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, SKOS, XSD

ROOT = Path(__file__).resolve().parents[1]
SABLE = Namespace("https://github.com/Acul01/SABLE/vocab#")
ID = Namespace("https://github.com/Acul01/SABLE/id/")
CITO = Namespace("http://purl.org/spar/cito/")
SCHEMA = Namespace("https://schema.org/")

LABELS = {"supported", "not supported", "unclear"}
EXCERPT = ("c001", "c002", "c006", "c010", "c021", "c062")

HEADER = """\
# Hypothesis graph for the SABLE runs.
# A claim is a hypothesis. The annotator's sentence is an evidence span.
# The paper is a DOI. The theme is the context node stored with the paper.
# cito:supports and cito:disputes follow the human label.
# sable:predictedLabel is the temperature-0 label.
# sable:decision is the one-round check.
# sable:tokenProbability is the first label token of the one-round check.
# sable:tokenEntropy is the mean token entropy of five full replies, averaged
# over the replies that emitted the majority label. sable:agreement is the
# share of those five replies.
# sable:labelChanged is present only where the span was deleted.

"""


@dataclass(frozen=True)
class Row:
    claim_id: str
    doi: str
    statement: str
    evidence: str
    gold: str
    theme: str
    title: str
    predicted: str
    evidence_in_topk: bool
    decision: str
    accepted_label: str | None
    token_probability: float | None
    quotes_ok: bool | None
    label_changed: bool | None
    label_after_deletion: str | None
    token_entropy: float | None = None
    agreement: float | None = None


def _csv(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def decisions_from_jsonl(path: Path) -> dict[str, dict]:
    """Decision lines only. Retrieve steps, including passage text, are dropped."""
    decisions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("tool") == "decision":
            decisions[row["claim_id"]] = row["output"]
    return decisions


def _label(value: str, claim_id: str) -> str:
    if value not in LABELS:
        raise ValueError(f"{claim_id}: unknown label {value!r}")
    return value


def _probability(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(value)
    return float(value)


def _decimal(value: float) -> Literal:
    text = format(value, ".16f").rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return Literal(text, datatype=XSD.decimal)


def _flag(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(value)
    return value == "True"


def make_row(claim: dict, source: dict, prediction: dict, intervention: dict | None, decision: dict, uncertainty: dict | None = None) -> Row:
    claim_id = claim["claim_id"]
    kept = decision.get("label")
    if decision["decision"] not in {"accept", "abstain"}:
        raise ValueError(f"{claim_id}: unknown decision {decision['decision']!r}")
    changed = None if intervention is None else _flag(intervention["flipped"])
    after = None if intervention is None else _label(intervention["predicted_without_evidence"], claim_id)
    quotes_ok = decision.get("quotes_ok")
    return Row(
        claim_id=claim_id,
        doi=claim["doi"],
        statement=claim["statement"],
        evidence=claim["evidence"],
        gold=_label(claim["label"], claim_id),
        theme=source["theme"],
        title=source["title"],
        predicted=_label(prediction["predicted"], claim_id),
        evidence_in_topk=_flag(prediction["evidence_in_topk"]),
        decision=decision["decision"],
        accepted_label=None if kept is None else _label(kept, claim_id),
        token_probability=_probability(decision.get("token_probability")),
        quotes_ok=None if quotes_ok is None else bool(quotes_ok),
        label_changed=changed,
        label_after_deletion=after,
        token_entropy=None if uncertainty is None else _probability(uncertainty["token_entropy"]),
        agreement=None if uncertainty is None else _probability(uncertainty["agreement"]),
    )


def load_rows(root: Path | None = None) -> list[Row]:
    root = ROOT if root is None else root
    sources = {row["doi"]: row for row in _csv(root / "data" / "SOURCES.csv")}
    predictions = {row["claim_id"]: row for row in _csv(root / "results" / "predictions.csv")}
    interventions = {row["claim_id"]: row for row in _csv(root / "results" / "intervention.csv")}
    decisions = decisions_from_jsonl(root / "results" / "v2" / "v2a.jsonl")
    uncertainty = {row["claim_id"]: row for row in _csv(root / "results" / "uncertainty.csv")}
    rows = []
    for claim in _csv(root / "data" / "CLAIMS.csv"):
        claim_id = claim["claim_id"]
        if claim_id not in decisions:
            raise ValueError(f"no one-round decision for {claim_id}")
        if claim_id not in uncertainty:
            raise ValueError(f"no five-draw uncertainty for {claim_id}")
        rows.append(make_row(
            claim,
            sources[claim["doi"]],
            predictions[claim_id],
            interventions.get(claim_id),
            decisions[claim_id],
            uncertainty[claim_id],
        ))
    return rows


def _paper(doi: str) -> URIRef:
    if not doi.startswith("10."):
        raise ValueError(f"expected a DOI, got {doi!r}")
    return URIRef(f"https://doi.org/{doi}")


def _bind(graph: Graph) -> None:
    graph.bind("sable", SABLE)
    graph.bind("id", ID)
    graph.bind("cito", CITO)
    graph.bind("dcterms", DCTERMS)
    graph.bind("prov", PROV)
    graph.bind("skos", SKOS)
    graph.bind("schema", SCHEMA)


def _vocabulary(graph: Graph) -> None:
    terms = {
        SABLE.Hypothesis: "hypothesis",
        SABLE.Evidence: "evidence span",
        SABLE.statement: "claim text",
        SABLE.span: "annotator span",
        SABLE.goldLabel: "human label",
        SABLE.predictedLabel: "temperature-0 label",
        SABLE.decision: "one-round decision",
        SABLE.acceptedLabel: "label kept by the one-round check",
        SABLE.tokenProbability: "probability of the first label token",
        SABLE.tokenEntropy: "mean token entropy of the majority replies",
        SABLE.agreement: "share of five replies with the majority label",
        SABLE.sampledBy: "five-reply sample",
        SABLE.evidenceFor: "evidence for",
        SABLE.evidenceInTopFive: "span was in the five chunks",
        SABLE.labelChanged: "deleting the span changed the temperature-0 label",
        SABLE.labelAfterDeletion: "temperature-0 label after the span was deleted",
        SABLE.quotesOk: "every quote was an exact substring",
        SABLE.predictedBy: "temperature-0 run",
        SABLE.checkedBy: "one-round run",
        SABLE.temperature: "decoding temperature",
    }
    for term, label in terms.items():
        kind = RDFS.Class if term in {SABLE.Hypothesis, SABLE.Evidence} else RDF.Property
        graph.add((term, RDF.type, kind))
        graph.add((term, RDFS.label, Literal(label)))
    graph.add((SABLE.tokenProbability, RDFS.comment, Literal(
        "Probability of the first label token in the one-round check."
    )))
    graph.add((SABLE.tokenEntropy, RDFS.comment, Literal(
        "Mean Shannon entropy, in bits, of every token in the completion. "
        "Each token uses its twenty most likely alternatives, and the remaining "
        "probability is one extra outcome. The token mean is averaged over the "
        "five replies that emitted the majority label."
    )))
    graph.add((SABLE.agreement, RDFS.comment, Literal(
        "Share of the five replies that emitted the majority label."
    )))
    graph.add((SABLE.labelChanged, RDFS.comment, Literal(
        "Present only where the evidence span was in the five chunks and was deleted."
    )))


def _activities(graph: Graph) -> tuple[URIRef, URIRef, URIRef]:
    model = ID["model/gpt-4o-mini"]
    temperature_0 = ID["run/temperature-0"]
    one_round = ID["run/v2a"]
    sampled = ID["run/sample-5"]
    graph.add((model, RDF.type, PROV.SoftwareAgent))
    graph.add((model, RDFS.label, Literal("gpt-4o-mini")))
    for activity, label, temperature in (
        (temperature_0, "temperature-0 label", "0.0"),
        (one_round, "one-round check", "0.0"),
        (sampled, "five full replies", "0.7"),
    ):
        graph.add((activity, RDF.type, PROV.Activity))
        graph.add((activity, RDFS.label, Literal(label)))
        graph.add((activity, PROV.wasAssociatedWith, model))
        graph.add((activity, SABLE.temperature, Literal(temperature, datatype=XSD.decimal)))
    return temperature_0, one_round, sampled


def _bool(value: bool) -> Literal:
    return Literal(value, datatype=XSD.boolean)


def build_graph(rows: list[Row]) -> Graph:
    graph = Graph()
    _bind(graph)
    _vocabulary(graph)
    temperature_0, one_round, sampled = _activities(graph)
    themes: dict[str, URIRef] = {}
    papers: set[URIRef] = set()
    for row in rows:
        hypothesis = ID[row.claim_id]
        evidence = ID[f"{row.claim_id}/evidence"]
        paper = _paper(row.doi)
        if row.theme not in themes:
            themes[row.theme] = ID[f"theme/{row.theme}"]
            graph.add((themes[row.theme], RDF.type, SKOS.Concept))
            graph.add((themes[row.theme], SKOS.prefLabel, Literal(row.theme)))
        theme = themes[row.theme]
        if paper not in papers:
            graph.add((paper, RDF.type, SCHEMA.ScholarlyArticle))
            graph.add((paper, DCTERMS.identifier, Literal(row.doi)))
            graph.add((paper, DCTERMS.title, Literal(row.title)))
            papers.add(paper)
        graph.add((paper, DCTERMS.subject, theme))
        graph.add((hypothesis, RDF.type, SABLE.Hypothesis))
        graph.add((hypothesis, SABLE.statement, Literal(row.statement)))
        graph.add((hypothesis, SABLE.goldLabel, Literal(row.gold)))
        graph.add((hypothesis, SABLE.predictedLabel, Literal(row.predicted)))
        graph.add((hypothesis, SABLE.decision, Literal(row.decision)))
        graph.add((hypothesis, SABLE.evidenceInTopFive, _bool(row.evidence_in_topk)))
        graph.add((hypothesis, DCTERMS.source, paper))
        graph.add((hypothesis, DCTERMS.subject, theme))
        graph.add((hypothesis, SABLE.predictedBy, temperature_0))
        graph.add((hypothesis, SABLE.checkedBy, one_round))
        graph.add((hypothesis, SABLE.sampledBy, sampled))
        if row.token_entropy is not None:
            graph.add((hypothesis, SABLE.tokenEntropy, _decimal(row.token_entropy)))
        if row.agreement is not None:
            graph.add((hypothesis, SABLE.agreement, _decimal(row.agreement)))
        if row.accepted_label is not None:
            graph.add((hypothesis, SABLE.acceptedLabel, Literal(row.accepted_label)))
        if row.token_probability is not None:
            graph.add((hypothesis, SABLE.tokenProbability, _decimal(row.token_probability)))
        if row.quotes_ok is not None:
            graph.add((hypothesis, SABLE.quotesOk, _bool(row.quotes_ok)))
        if row.label_changed is not None:
            graph.add((hypothesis, SABLE.labelChanged, _bool(row.label_changed)))
            graph.add((hypothesis, SABLE.labelAfterDeletion, Literal(row.label_after_deletion)))
        graph.add((evidence, RDF.type, SABLE.Evidence))
        graph.add((evidence, SABLE.span, Literal(row.evidence)))
        graph.add((evidence, SABLE.evidenceFor, hypothesis))
        graph.add((evidence, DCTERMS.source, paper))
        if row.gold == "supported":
            graph.add((evidence, CITO.supports, hypothesis))
        elif row.gold == "not supported":
            graph.add((evidence, CITO.disputes, hypothesis))
    return graph


def excerpt_mermaid(rows: list[Row]) -> str:
    """Six hypotheses, their spans, papers, and themes."""
    chosen = [row for row in rows if row.claim_id in EXCERPT]
    lines = ["flowchart LR"]
    papers: dict[str, str] = {}
    themes: dict[str, str] = {}
    for row in chosen:
        number = row.claim_id[1:]
        hypothesis = f"h{number}"
        evidence = f"e{number}"
        paper = "p" + re.sub(r"[^A-Za-z0-9]", "", row.doi)
        theme = "t" + row.theme
        changed = " · label changed" if row.label_changed else ""
        lines.append(f'  {hypothesis}["{row.claim_id} · {row.gold}{changed}"]')
        lines.append(f'  {evidence}["{row.claim_id} span"]')
        papers.setdefault(paper, row.doi)
        themes.setdefault(theme, row.theme)
        if row.gold == "supported":
            relation = "supports"
        elif row.gold == "not supported":
            relation = "disputes"
        else:
            relation = "evidence for"
        lines.append(f"  {evidence} -->|{relation}| {hypothesis}")
        lines.append(f"  {evidence} -->|from| {paper}")
        lines.append(f"  {hypothesis} -->|about| {theme}")
    for node, doi in papers.items():
        lines.append(f'  {paper_node(node, doi)}')
    for node, theme in themes.items():
        lines.append(f'  {node}["{theme}"]')
    return "\n".join(lines)


def paper_node(node: str, doi: str) -> str:
    return f'{node}["{doi}"]'


def write_graph(path: Path | None = None, rows: list[Row] | None = None) -> Graph:
    path = ROOT / "graph" / "hypotheses.ttl" if path is None else path
    rows = load_rows() if rows is None else rows
    graph = build_graph(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + graph.serialize(format="turtle"), encoding="utf-8")
    return graph


def main() -> int:
    rows = load_rows()
    write_graph(rows=rows)
    print(f"{len(rows)} hypotheses -> graph/hypotheses.ttl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

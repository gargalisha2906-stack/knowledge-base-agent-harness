"""Routing: deciding which part of the knowledge base a query belongs to.

Routing narrows retrieval to the right domain's documents before vector
search runs. That both improves precision (a billing question never matches
a security doc) and mirrors how support teams actually organise knowledge.

The router here is transparent keyword scoring - easy to read, easy to
debug, and honest about confidence. Low confidence routes to "other",
which searches the whole KB; very low confidence escalates to a human.
The seam: swap ``Router`` for an LLM classifier without touching the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .embeddings import tokenize


@dataclass(frozen=True)
class Domain:
    """A support domain and the documents that belong to it."""

    name: str
    keywords: tuple[str, ...]
    doc_ids: tuple[str, ...]


@dataclass
class RouteDecision:
    domain: str            # domain name, or "other"
    confidence: float      # 0..1
    doc_ids: list[str]     # documents retrieval is allowed to touch
    scores: dict[str, int] = field(default_factory=dict)  # per-domain raw hits


class Router(Protocol):
    def route(self, query: str) -> RouteDecision: ...


class KeywordRouter:
    """Score each domain by keyword hits; confidence = top score mass."""

    def __init__(self, domains: list[Domain], min_confidence: float = 0.35):
        self.domains = domains
        self.min_confidence = min_confidence

    def route(self, query: str) -> RouteDecision:
        tokens = set(tokenize(query))
        lowered = query.lower()
        def hits(keywords: tuple[str, ...]) -> int:
            # Single-word keywords match whole tokens only (so "tax" does not
            # fire on "taxes" or "data" on "databases"); multi-word keyword
            # phrases match as substrings.
            return sum(
                1 for kw in keywords
                if (kw in lowered if " " in kw else kw in tokens)
            )
        scores = {d.name: hits(d.keywords) for d in self.domains}
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best] / total if total else 0.0
        if scores[best] == 0 or confidence < self.min_confidence:
            return RouteDecision(
                domain="other",
                confidence=confidence,
                doc_ids=[d for dom in self.domains for d in dom.doc_ids],
                scores=scores,
            )
        domain = next(d for d in self.domains if d.name == best)
        return RouteDecision(best, confidence, list(domain.doc_ids), scores)


# The domains of the demo knowledge base. In a real deployment these come
# from config so support ops can add a domain without a code change.
DOMAINS: list[Domain] = [
    Domain("refunds", ("refund", "refunds", "return", "money back", "cancel",
                       "cancelled", "chargeback", "charged twice", "duplicate charge", "refunding",
                       "duplicate", "refunded", "pro-rata", "prorated refund"),
           ("refund_policy.md",)),
    Domain("billing", ("invoice", "invoices", "payment", "card", "price", "pricing",
                       "plan", "subscription", "billing", "receipt", "tax", "gst",
                       "vat", "cost", "seats", "trial", "discount", "nonprofit",
                       "nonprofits", "paypal", "upgrade", "downgrade", "checkout",
                       "failed payment", "charge", "charged"),
           ("billing_faq.md", "plans.md")),
    Domain("technical", ("error", "bug", "login", "log in", "password", "api",
                         "timeout", "crash", "slow", "integration", "webhook",
                         "webhooks", "sdk", "import", "csv", "pagination",
                         "paginate", "endpoint", "endpoints", "429", "signature",
                         "rate limit", "2fa", "codes", "reset link", "cache",
                         "dashboard", "row", "token", "retry", "retries"),
           ("troubleshooting.md", "api_guide.md")),
    Domain("account", ("account", "profile", "delete", "privacy",
                       "security", "2fa", "two-factor", "password", "email", "team", "teammate",
                       "teammates", "invite", "invites", "role", "roles", "sms",
                       "workspace", "export", "device", "sign-in", "sessions",
                       "backup codes", "admin", "member", "viewer"),
           ("account_security.md",)),
]

"""Relationship builder.

Deterministic utilities for constructing inter-issue graphs from a
collection of ``NormalizedIssue`` records.  Every function is pure
(no I/O, no LLM calls) and operates over in-memory lists.

Public API
----------
- ``build_epic_hierarchy(issues)``   → epic-key → child-keys mapping.
- ``build_dependency_graph(issues)`` → nodes / edges / blocked chains.
- ``group_by_component(issues)``     → component → issue-keys mapping.
- ``find_related_clusters(issues)``  → groups of related issue keys.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from jira_contextualization.models.normalized_issue import NormalizedIssue

# Link-types treated as hard dependency / blocker edges
_BLOCKING_LINK_TYPES = {
    "blocks",
    "depends",
    "depends on",
    "dependency",
    "defect",
    "is blocked by",
}


def build_epic_hierarchy(issues: list[NormalizedIssue]) -> dict[str, list[str]]:
    """Build a mapping of epic keys to their child issue keys.

    An issue is considered a child of an epic when its ``epic_link``
    field points to the epic's key.

    Args:
        issues: Parsed Jira issues.

    Returns:
        ``{epic_key: [child_key, …]}`` — epics with no children are
        omitted.
    """
    hierarchy: dict[str, list[str]] = defaultdict(list)

    # Index issue keys so we know which keys are actually epics
    issue_keys: set[str] = {i.issue_key for i in issues}

    for issue in issues:
        if issue.epic_link:
            hierarchy[issue.epic_link].append(issue.issue_key)

    # Sort children for deterministic output
    return {k: sorted(v) for k, v in sorted(hierarchy.items())}


def build_dependency_graph(issues: list[NormalizedIssue]) -> dict[str, Any]:
    """Construct a dependency graph from issue links.

    Only link types recognised as *blocking* relationships are included
    (see ``_BLOCKING_LINK_TYPES``).  The graph includes:

    - **nodes**: all issue keys that participate in at least one edge.
    - **edges**: ``[{source, target, type}, …]`` (``source`` blocks
      ``target``).
    - **blocked_chains**: maximal chains where each node blocks the next,
      found via DFS.

    Args:
        issues: Parsed Jira issues.

    Returns:
        Dictionary with ``nodes``, ``edges``, and ``blocked_chains``.
    """
    nodes: set[str] = set()
    edges: list[dict[str, str]] = []

    # Adjacency list for chain detection (source → [targets])
    adj: dict[str, list[str]] = defaultdict(list)

    for issue in issues:
        for link in issue.issue_links:
            lt = link.link_type.lower().strip()
            if lt not in _BLOCKING_LINK_TYPES:
                continue

            # Determine directionality:
            # "outward" + "Blocks" means *this issue* blocks the target
            # "inward" + "Blocks" means *the target* blocks this issue
            if link.direction == "outward":
                source, target = issue.issue_key, link.target_key
            else:
                source, target = link.target_key, issue.issue_key

            nodes.add(source)
            nodes.add(target)
            edges.append({"source": source, "target": target, "type": link.link_type})
            adj[source].append(target)

    blocked_chains = _find_maximal_chains(adj)

    return {
        "nodes": sorted(nodes),
        "edges": edges,
        "blocked_chains": blocked_chains,
    }


def group_by_component(issues: list[NormalizedIssue]) -> dict[str, list[str]]:
    """Group issue keys by their component names.

    Issues with multiple components appear under each component.  Issues
    with *no* components are collected under the key ``'_unassigned'``.

    Args:
        issues: Parsed Jira issues.

    Returns:
        ``{component_name: [issue_key, …]}``.
    """
    groups: dict[str, list[str]] = defaultdict(list)

    for issue in issues:
        if issue.components:
            for comp in issue.components:
                groups[comp].append(issue.issue_key)
        else:
            groups["_unassigned"].append(issue.issue_key)

    return {k: sorted(v) for k, v in sorted(groups.items())}


def find_related_clusters(issues: list[NormalizedIssue]) -> list[list[str]]:
    """Find groups of transitively related issues.

    Two issues are *related* if they share **any** issue-link (regardless
    of type or direction).  The function computes connected components
    using union–find, then returns each component as a sorted list of
    issue keys.  Singleton clusters (issues with no links) are excluded.

    Args:
        issues: Parsed Jira issues.

    Returns:
        List of clusters, each a sorted list of issue keys.  Sorted by
        cluster size (largest first).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        """Path-compressed find."""
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build unions from all link relationships
    for issue in issues:
        for link in issue.issue_links:
            union(issue.issue_key, link.target_key)

        # Also union via epic link
        if issue.epic_link:
            union(issue.issue_key, issue.epic_link)

    # Collect clusters
    clusters: dict[str, list[str]] = defaultdict(list)
    all_keys = set(parent.keys())
    for key in all_keys:
        root = find(key)
        clusters[root].append(key)

    # Filter singletons, sort by descending size
    result = [
        sorted(members)
        for members in clusters.values()
        if len(members) > 1
    ]
    result.sort(key=len, reverse=True)
    return result


# ── Private helpers ──────────────────────────────────────────────────────


def _find_maximal_chains(adj: dict[str, list[str]]) -> list[list[str]]:
    """Find maximal blocked chains using iterative DFS.

    A chain is a path ``[A, B, C, …]`` where A blocks B, B blocks C, etc.
    We start DFS from every node that has no incoming edges (a "root
    blocker") and track the longest paths.

    Returns:
        Chains of length ≥ 2, sorted by length (longest first).
    """
    # Identify nodes with incoming edges
    has_incoming: set[str] = set()
    for targets in adj.values():
        has_incoming.update(targets)

    # Start nodes = nodes with outgoing edges but no incoming
    all_nodes = set(adj.keys())
    start_nodes = all_nodes - has_incoming
    if not start_nodes:
        # If everything has incoming edges (cycle), start everywhere
        start_nodes = all_nodes

    chains: list[list[str]] = []

    for start in sorted(start_nodes):
        # Iterative DFS using an explicit stack of (node, path)
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            children = adj.get(node, [])
            if not children:
                # Leaf — record chain if length ≥ 2
                if len(path) >= 2:
                    chains.append(path)
            else:
                for child in children:
                    if child not in path:  # cycle guard
                        stack.append((child, [*path, child]))

    # Deduplicate (same chain reachable from different starts) and sort
    unique: dict[str, list[str]] = {}
    for chain in chains:
        key = "→".join(chain)
        if key not in unique:
            unique[key] = chain

    result = list(unique.values())
    result.sort(key=len, reverse=True)
    return result

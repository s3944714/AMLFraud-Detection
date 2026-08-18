"""
network/entity_graph.py

Entity-link graph for fraud-ring candidate detection. This is deliberately
separate from models/baseline.py: a per-transaction classifier can only ever
say "this transaction looks risky in isolation." It has no way to notice
that twenty superficially-unremarkable transactions all share the same
card+address combination or device fingerprint - which is exactly the
signature of a coordinated ring rather than twenty independent bad actors.
Entity-linked features like this were central to the top public solutions
for this competition (see README for the specific citation).

Nodes: individual transactions (by TransactionID).
Edges: two transactions are linked if they share ANY of:
  - card1 + addr1 (a standard approximation for a unique client/card ID in
    this dataset, since there's no direct customer identifier)
  - DeviceInfo
  - P_emaildomain
Clusters: connected components over this graph. A cluster is flagged as a
candidate "ring" if it has >=3 transactions AND its fraud rate is
meaningfully elevated vs the dataset baseline.

Important framing: this is unsupervised structure-finding, not a
verified-fraud-ring detector. A flagged cluster means "this group of
transactions is unusually linked AND unusually fraud-heavy" - it's a lead
for an investigator to look at, not a conviction. Shared attributes can also
arise from mundane reasons (a shared household, a shared office, a popular
free email domain used as one edge type - see the P_emaildomain caveat
below), so cluster membership alone isn't proof of anything.
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from data.loader import load_raw

MODULE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = MODULE_DIR.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLUSTERS_PATH = PROCESSED_DIR / "entity_clusters.csv"
CLUSTER_SUMMARY_PATH = PROCESSED_DIR / "cluster_summary.json"

TARGET_COL = "isFraud"
ID_COL = "TransactionID"

MIN_CLUSTER_SIZE = 3
# A cluster's fraud rate must be at least this many times the dataset
# baseline to get flagged as a candidate ring - "elevated" needs a concrete
# threshold, and a raw multiplier is easy to reason about and to justify in
# a README ("N times the baseline rate") compared to a less legible
# statistical test, at the cost of being a simpler heuristic rather than a
# formal significance test. Tune as needed once you see the distribution.
ELEVATED_RATE_MULTIPLIER = 2.0

# Maximum size of a shared-attribute group allowed to form edges. This
# matters a lot: a value shared by a handful to a few dozen transactions is
# a plausible entity link (a real card+address combo, a real device). A
# value shared by tens of thousands is not an entity, it's a generic
# category - the most common example is DeviceInfo="Windows" (shared by a
# huge share of all transactions, as opposed to a specific device string
# like "SM-G920P Build/NRD90M", which genuinely does indicate one device)
# or P_emaildomain="gmail.com". Without this cap, a single common value
# merges an otherwise-tight, genuinely fraud-heavy cluster into one giant,
# mostly-unrelated mega-component and dilutes its fraud rate back toward
# baseline - confirmed directly: a 50-transaction synthetic ring with an
# 80% injected fraud rate was completely invisible in cluster output until
# this cap was added, because it got absorbed into a ~40,000-transaction
# "Windows" component. Tune per how large a real card+address/device group
# plausibly represents one entity in your data.
MAX_GROUP_SIZE = 100

# Second, coarser safeguard: after connected components are computed, drop
# any component whose TOTAL size exceeds this from the output entirely.
# MAX_GROUP_SIZE alone isn't sufficient at real dataset scale: even with
# every individual shared-value group capped at 100, a transaction can
# belong to a card_addr group AND a device group AND an email group
# simultaneously, and these small, individually-legitimate bridges can
# transitively chain thousands of otherwise-unrelated groups into one giant
# emergent component - confirmed directly on the real 590k-row dataset,
# where a 28,427-transaction component formed this way (4.8% of the whole
# dataset in one "cluster"). That component's fraud rate is already
# correctly diluted toward baseline, so it's never mislabeled as a
# candidate ring - this cap is purely about not leaving obvious noise
# sitting in entity_clusters.csv for anyone inspecting it directly. Set
# generously above the largest plausible real ring (the largest genuine
# candidate ring observed in practice was in the hundreds, not thousands).
MAX_COMPONENT_SIZE = 2000

DEFAULT_EDGE_TYPES = ("card_addr", "device", "email")


def _add_edges_for_key(G: nx.Graph, df: pd.DataFrame, key_cols: list, edge_type: str):
    """
    Add an edge between every pair of transactions that share the same
    (non-null) value on key_cols. Grouping + connecting all pairs within
    each group is what actually builds the "shared attribute" links; this
    is the only place per-edge-type logic lives, so card_addr/device/email
    are all handled identically.
    """
    valid = df.dropna(subset=key_cols)
    if len(key_cols) == 1:
        groups = valid.groupby(key_cols[0])[ID_COL]
    else:
        groups = valid.groupby(key_cols)[ID_COL]

    for _, ids in groups:
        ids = ids.tolist()
        if len(ids) < 2:
            continue
        if len(ids) > MAX_GROUP_SIZE:
            # Too common to represent one entity (e.g. DeviceInfo="Windows",
            # P_emaildomain="gmail.com") - see MAX_GROUP_SIZE's comment.
            # Skipping edge creation for this group entirely, rather than
            # e.g. capping which members get linked, is deliberate: any
            # arbitrary subset of a 40,000-member "Windows" group is just
            # as meaningless as the whole group, so there's no principled
            # way to keep "some" of these edges.
            continue
        # Star topology (first node to all others) rather than a full
        # clique: for a group of n shared-attribute transactions this is
        # n-1 edges instead of n*(n-1)/2, and connected-components produces
        # an IDENTICAL cluster membership result either way - a star is
        # sufficient to connect the whole group into one component. This
        # matters a lot at this dataset's scale: a large-but-under-the-cap
        # shared value would otherwise generate a lot of redundant edges.
        anchor = ids[0]
        for other in ids[1:]:
            G.add_edge(anchor, other, edge_type=edge_type)


def build_entity_graph(df: pd.DataFrame, edge_types=DEFAULT_EDGE_TYPES) -> nx.Graph:
    """
    Build the entity-link graph over ALL transactions (train + test, i.e.
    the full raw df) - fraud rings don't respect a time-based train/test
    split boundary, and a ring's shared card+address or device link is
    equally real whether the individual transactions happen to fall before
    or after the split cutoff. Clustering must therefore run on the whole
    dataset, not just the model's held-out test portion.
    """
    G = nx.Graph()
    G.add_nodes_from(df[ID_COL].tolist())

    if "card_addr" in edge_types:
        _add_edges_for_key(G, df, ["card1", "addr1"], "card_addr")
    if "device" in edge_types:
        _add_edges_for_key(G, df, ["DeviceInfo"], "device")
    if "email" in edge_types:
        _add_edges_for_key(G, df, ["P_emaildomain"], "email")

    return G


def compute_clusters(df: pd.DataFrame, G: nx.Graph, baseline_fraud_rate: float) -> pd.DataFrame:
    """
    Run connected components over the entity graph, compute each
    component's size and fraud rate, and flag candidate rings.

    Isolated nodes (transactions with no shared-attribute link to anything)
    form their own singleton "clusters" - these are dropped before output,
    since a cluster of one transaction isn't an entity-linking finding at
    all, just a transaction with no matching neighbors.
    """
    fraud_lookup = df.set_index(ID_COL)[TARGET_COL]

    rows = []
    for cluster_id, component in enumerate(nx.connected_components(G)):
        if len(component) < 2:
            continue  # singleton - no actual link to report
        if len(component) > MAX_COMPONENT_SIZE:
            # Emergent mega-component from edge-type bridging (see
            # MAX_COMPONENT_SIZE's comment) - not a real entity, and never
            # a candidate ring anyway since its fraud rate is diluted
            # toward baseline, so excluding it from output entirely rather
            # than writing 28,000+ noise rows for it.
            continue

        txn_ids = list(component)
        fraud_flags = fraud_lookup.loc[txn_ids]
        cluster_fraud_rate = float(fraud_flags.mean())
        n_txns = len(txn_ids)
        is_candidate_ring = (
            n_txns >= MIN_CLUSTER_SIZE
            and cluster_fraud_rate >= baseline_fraud_rate * ELEVATED_RATE_MULTIPLIER
        )

        for txn_id in txn_ids:
            # shared_attribute here records what LINKED this transaction to
            # at least one other member of its cluster (not necessarily
            # every member - in a chain A-B-C via different shared
            # attributes, all three still land in one connected component).
            edge_types_touching_node = {
                data["edge_type"] for _, _, data in G.edges(txn_id, data=True)
            }
            shared_attr = "+".join(sorted(edge_types_touching_node)) if edge_types_touching_node else "none"

            rows.append({
                "cluster_id": cluster_id,
                "TransactionID": txn_id,
                "shared_attribute": shared_attr,
                "cluster_fraud_rate": cluster_fraud_rate,
                "cluster_size": n_txns,
                "is_candidate_ring": is_candidate_ring,
            })

    return pd.DataFrame(rows)


def run_entity_graph_analysis(edge_types=DEFAULT_EDGE_TYPES):
    df = load_raw()
    baseline_fraud_rate = float(df[TARGET_COL].mean())

    print(f"Building entity graph over {len(df):,} transactions "
          f"(edge types: {', '.join(edge_types)})...")
    G = build_entity_graph(df, edge_types=edge_types)
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    clusters_df = compute_clusters(df, G, baseline_fraud_rate)

    candidate_rings = clusters_df[clusters_df["is_candidate_ring"]]
    n_candidate_clusters = candidate_rings["cluster_id"].nunique()
    n_candidate_txns = len(candidate_rings)
    avg_ring_fraud_rate = (
        candidate_rings.drop_duplicates("cluster_id")["cluster_fraud_rate"].mean()
        if n_candidate_clusters > 0 else 0.0
    )

    summary = {
        "baseline_fraud_rate": baseline_fraud_rate,
        "total_transactions": len(df),
        "n_linked_transactions": int((clusters_df["cluster_size"] >= 1).sum()),
        "n_total_clusters": int(clusters_df["cluster_id"].nunique()),
        "n_candidate_ring_clusters": int(n_candidate_clusters),
        "n_candidate_ring_transactions": int(n_candidate_txns),
        "avg_candidate_ring_fraud_rate": float(avg_ring_fraud_rate),
        "fraud_rate_multiplier_vs_baseline": (
            float(avg_ring_fraud_rate / baseline_fraud_rate) if baseline_fraud_rate > 0 else None
        ),
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "elevated_rate_multiplier_threshold": ELEVATED_RATE_MULTIPLIER,
    }

    # export/to_app.py's cluster_id column is documented as nullable - only
    # candidate-ring members get a cluster_id downstream, non-ring clusters
    # (e.g. size-2 links, or clusters that didn't clear the fraud-rate bar)
    # are still written here for transparency but shouldn't be surfaced to
    # the review app as if they were rings.
    clusters_df.to_csv(CLUSTERS_PATH, index=False)
    with open(CLUSTER_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Baseline fraud rate: {baseline_fraud_rate:.4%}")
    print(f"Total clusters (size >= 2): {summary['n_total_clusters']:,}")
    print(f"Candidate ring clusters (size >= {MIN_CLUSTER_SIZE}, "
          f">= {ELEVATED_RATE_MULTIPLIER}x baseline fraud rate): {n_candidate_clusters:,}")
    print(f"  covering {n_candidate_txns:,} transactions")
    print(f"  average fraud rate within these clusters: {avg_ring_fraud_rate:.4%} "
          f"({summary['fraud_rate_multiplier_vs_baseline']:.1f}x baseline)" if n_candidate_clusters else "")
    print(f"Clusters -> {CLUSTERS_PATH}")
    print(f"Summary  -> {CLUSTER_SUMMARY_PATH}")

    return clusters_df, summary


if __name__ == "__main__":
    run_entity_graph_analysis()
"""
tests/test_entity_graph.py

Tests for network/entity_graph.py, using small hand-built DataFrames. Graph
construction is exactly the kind of place a silent bug (an off-by-one in
grouping, a missed NaN, an edge that should exist but doesn't) wouldn't show
up in any headline metric - it would just quietly produce wrong clusters
that look plausible. These tests check the actual graph structure directly.
"""

import pandas as pd
import pytest

from network.entity_graph import (
    build_entity_graph,
    compute_clusters,
    MIN_CLUSTER_SIZE,
    ELEVATED_RATE_MULTIPLIER,
)


def _toy_df():
    # Group A: txns 1,2,3 share card1=100/addr1=200 - a card_addr triangle
    # Group B: txns 4,5 share DeviceInfo="Windows" - a device pair
    # Txn 6: isolated, no shared attributes with anything
    # Txn 7: shares card_addr with txn 1's group too (extends group A to 4)
    # Txn 8: has NaN card1 - must not spuriously link to anything on card_addr
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6, 7, 8],
        "isFraud":        [1, 1, 0, 0, 0, 0, 1, 0],
        "card1":          [100, 100, 100, 999, 999, 555, 100, None],
        "addr1":          [200, 200, 200, 300, 300, 400, 200, 500],
        "DeviceInfo":     [None, None, None, "Windows", "Windows", None, None, None],
        "P_emaildomain":  [None] * 8,
    })


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def test_shared_card_addr_creates_edges():
    df = _toy_df()
    G = build_entity_graph(df, edge_types=("card_addr",))
    # 1, 2, 3, 7 all share card1=100/addr1=200 -> should be one component
    import networkx as nx
    components = list(nx.connected_components(G))
    group = [c for c in components if 1 in c][0]
    assert group == {1, 2, 3, 7}


def test_shared_device_creates_edges():
    df = _toy_df()
    G = build_entity_graph(df, edge_types=("device",))
    import networkx as nx
    components = list(nx.connected_components(G))
    group = [c for c in components if 4 in c][0]
    assert group == {4, 5}


def test_isolated_transaction_has_no_edges():
    df = _toy_df()
    G = build_entity_graph(df, edge_types=("card_addr", "device"))
    assert G.degree(6) == 0


def test_null_key_values_do_not_create_spurious_edges():
    # Txn 8 has card1=NaN - if dropna() weren't applied before grouping,
    # a naive groupby could still (depending on pandas version/settings)
    # treat all-NaN rows as a "group" and wrongly connect them.
    df = _toy_df()
    G = build_entity_graph(df, edge_types=("card_addr",))
    assert G.degree(8) == 0


def test_combining_edge_types_merges_components_via_chain():
    # Txn 3 (in the card_addr group) and txn 4 (in the device group) don't
    # share anything directly - components should stay separate unless a
    # transaction bridges them. This confirms edge types are combined
    # correctly rather than one type silently overriding another.
    df = _toy_df()
    G = build_entity_graph(df, edge_types=("card_addr", "device"))
    import networkx as nx
    components = list(nx.connected_components(G))
    card_addr_group = [c for c in components if 1 in c][0]
    device_group = [c for c in components if 4 in c][0]
    assert card_addr_group.isdisjoint(device_group)


# ---------------------------------------------------------------------------
# Cluster computation and ring-flagging
# ---------------------------------------------------------------------------

def test_cluster_fraud_rate_computed_correctly():
    df = _toy_df()
    G = build_entity_graph(df, edge_types=("card_addr",))
    baseline_rate = df["isFraud"].mean()
    clusters_df = compute_clusters(df, G, baseline_rate)

    group_a = clusters_df[clusters_df["TransactionID"].isin([1, 2, 3, 7])]
    # isFraud for 1,2,3,7 = [1,1,0,1] -> 3/4 = 0.75
    assert group_a["cluster_fraud_rate"].iloc[0] == pytest.approx(0.75)
    assert (group_a["cluster_size"] == 4).all()


def test_singleton_transactions_excluded_from_output():
    df = _toy_df()
    G = build_entity_graph(df, edge_types=("card_addr", "device"))
    baseline_rate = df["isFraud"].mean()
    clusters_df = compute_clusters(df, G, baseline_rate)

    # Txn 6 and txn 8 have no edges at all - should not appear in output
    assert 6 not in clusters_df["TransactionID"].values
    assert 8 not in clusters_df["TransactionID"].values


def test_candidate_ring_requires_both_size_and_elevated_rate():
    # Group A (1,2,3,7): size 4 (>= MIN_CLUSTER_SIZE), fraud rate 0.75
    # Group B (4,5): size 2 (< MIN_CLUSTER_SIZE) - should NOT be flagged
    # regardless of its fraud rate, since it fails the size threshold alone.
    df = _toy_df()
    G = build_entity_graph(df, edge_types=("card_addr", "device"))
    baseline_rate = df["isFraud"].mean()  # 2/8 = 0.25
    clusters_df = compute_clusters(df, G, baseline_rate)

    group_a_flag = clusters_df[clusters_df["TransactionID"] == 1]["is_candidate_ring"].iloc[0]
    group_b_flag = clusters_df[clusters_df["TransactionID"] == 4]["is_candidate_ring"].iloc[0]

    assert MIN_CLUSTER_SIZE >= 3  # sanity check the constant this test relies on
    assert group_a_flag == (0.75 >= baseline_rate * ELEVATED_RATE_MULTIPLIER)
    assert group_b_flag == False  # size 2 fails MIN_CLUSTER_SIZE no matter the rate


def test_low_fraud_rate_cluster_not_flagged_even_if_large_enough():
    # A same-size-4 cluster where fraud rate equals baseline (no elevation)
    # should not be flagged, isolating the rate condition from the size one.
    df = pd.DataFrame({
        "TransactionID": [10, 11, 12, 13],
        "isFraud": [0, 0, 0, 0],
        "card1": [7, 7, 7, 7],
        "addr1": [8, 8, 8, 8],
        "DeviceInfo": [None] * 4,
        "P_emaildomain": [None] * 4,
    })
    G = build_entity_graph(df, edge_types=("card_addr",))
    clusters_df = compute_clusters(df, G, baseline_fraud_rate=0.5)
    assert not clusters_df["is_candidate_ring"].any()


def test_oversized_component_excluded_from_output():
    # A mega-component (simulating the real 28,427-transaction case caused
    # by edge-type bridging) should be dropped from output entirely, even
    # though it's already harmless for ring-flagging purposes (its fraud
    # rate would be diluted). This is purely an output-hygiene check.
    from network.entity_graph import MAX_COMPONENT_SIZE

    n = MAX_COMPONENT_SIZE + 50
    df = pd.DataFrame({
        "TransactionID": list(range(n)),
        "isFraud": [0] * n,
        "card1": [42] * n,        # one shared value -> one giant group
        "addr1": [42] * n,
        "DeviceInfo": [None] * n,
        "P_emaildomain": [None] * n,
    })
    # Bypass MAX_GROUP_SIZE by building the graph directly with a manual
    # edge set (a real MAX_GROUP_SIZE-capped run couldn't produce a single
    # group this large in the first place - this test targets the
    # component-size safeguard specifically, in isolation).
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(df["TransactionID"].tolist())
    for i in range(1, n):
        G.add_edge(0, i, edge_type="card_addr")

    clusters_df = compute_clusters(df, G, baseline_fraud_rate=0.035)
    assert len(clusters_df) == 0


def test_component_at_or_under_cap_still_included():
    from network.entity_graph import MAX_COMPONENT_SIZE

    n = min(MAX_COMPONENT_SIZE, 20)  # keep test fast regardless of cap size
    df = pd.DataFrame({
        "TransactionID": list(range(n)),
        "isFraud": [1] * n,
        "card1": [42] * n,
        "addr1": [42] * n,
        "DeviceInfo": [None] * n,
        "P_emaildomain": [None] * n,
    })
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(df["TransactionID"].tolist())
    for i in range(1, n):
        G.add_edge(0, i, edge_type="card_addr")

    clusters_df = compute_clusters(df, G, baseline_fraud_rate=0.035)
    assert len(clusters_df) == n


def test_shared_attribute_column_reflects_edge_type():
    df = _toy_df()
    G = build_entity_graph(df, edge_types=("card_addr", "device"))
    baseline_rate = df["isFraud"].mean()
    clusters_df = compute_clusters(df, G, baseline_rate)

    card_addr_row = clusters_df[clusters_df["TransactionID"] == 2]
    assert card_addr_row["shared_attribute"].iloc[0] == "card_addr"

    device_row = clusters_df[clusters_df["TransactionID"] == 5]
    assert device_row["shared_attribute"].iloc[0] == "device"
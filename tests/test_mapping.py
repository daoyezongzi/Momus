from critic.mapping import make_random_mapping, mapping_coverage, snapshot_mapping


def test_mapping_hash_is_order_independent():
    first = snapshot_mapping(
        {"a": ["b", "c"]},
        {"a": "x", "b": "x", "c": "x"},
        ["a", "b", "c"],
        "v1",
    )
    second = snapshot_mapping(
        {"a": ["c", "b"]},
        {"c": "x", "b": "x", "a": "x"},
        ["c", "b", "a"],
        "v1",
    )
    assert first.sha256 == second.sha256


def test_random_mapping_is_deterministic_per_fold():
    snapshot = snapshot_mapping(
        {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]},
        {"a": "x", "b": "x", "c": "y", "d": "y"},
        ["a", "b", "c", "d"],
        "v1",
    )
    assert make_random_mapping(snapshot, "fold_2023", 3) == make_random_mapping(
        snapshot, "fold_2023", 3
    )


def test_mapping_coverage_reports_unmapped_codes_without_filling_them():
    snapshot = snapshot_mapping(
        {"a": ["b"]},
        {"a": "x", "b": "x", "c": "y"},
        ["a", "b", "c"],
        "v1",
    )
    coverage = mapping_coverage(snapshot)
    assert coverage["n_codes"] == 3
    assert coverage["n_mapped_targets"] == 1
    assert coverage["unmapped"] == ["b", "c"]

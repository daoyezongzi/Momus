from dataclasses import dataclass
import hashlib
import json
import random


@dataclass(frozen=True)
class ChainMappingSnapshot:
    version: str
    mapping: dict[str, tuple[str, ...]]
    chain_by_code: dict[str, str]
    stock_pool: tuple[str, ...]
    sha256: str


def snapshot_mapping(mapping, chain_by_code, stock_pool, version):
    canonical_mapping = {
        code: tuple(sorted(related)) for code, related in sorted(mapping.items())
    }
    payload = {
        "version": version,
        "mapping": {code: list(related) for code, related in canonical_mapping.items()},
        "chain_by_code": dict(sorted(chain_by_code.items())),
        "stock_pool": sorted(stock_pool),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return ChainMappingSnapshot(
        version=version,
        mapping=canonical_mapping,
        chain_by_code=dict(sorted(chain_by_code.items())),
        stock_pool=tuple(sorted(stock_pool)),
        sha256=digest,
    )


def make_random_mapping(snapshot: ChainMappingSnapshot, fold_id: str, seed: int):
    derived_seed = int(hashlib.sha256(f"{fold_id}:{seed}".encode()).hexdigest()[:16], 16)
    rng = random.Random(derived_seed)
    pool = list(snapshot.stock_pool)
    result: dict[str, list[str]] = {}
    for target, related in snapshot.mapping.items():
        candidates = [code for code in pool if code != target]
        rng.shuffle(candidates)
        result[target] = candidates[: len(related)]
    return result


def mapping_coverage(snapshot: ChainMappingSnapshot) -> dict[str, object]:
    mapped = set(snapshot.mapping)
    pool = set(snapshot.stock_pool)
    return {
        "n_codes": len(pool),
        "n_mapped_targets": len(mapped & pool),
        "unmapped": sorted(pool - mapped),
    }

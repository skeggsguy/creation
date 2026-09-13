"""Stage 5 — carve validation sets and plan the mix.

Reads every ``data/clean/<domain>/`` directory, splits a validation set off each
domain, works out how many epochs each themed domain needs in order to hit its
ratio from ``DataConfig.mix``, and writes:

  data/clean/manifest.json   ordered [{file, domain, split, epochs, est_tokens, docs}]
  data/clean/stats.md        the same thing for humans

It deliberately does **not** concatenate or shuffle anything — the tokenizer
stage consumes the manifest and does that.

Validation carve
----------------
``val_tokens_per_domain`` (2M by default) is taken from the *tail* of each domain,
by rewriting the last shard(s) into ``_val_<domain>.jsonl`` (excluded from train
by the leading underscore) plus ``<domain>_rest.jsonl``.  The split is recorded in
``<domain>/_valsplit.json`` and skipped on re-runs.

Epoch planning
--------------
``epochs[d] = clamp(target_tokens[d] / available_tokens[d], .., max_domain_epochs)``.
Fractional epochs below 1.0 are legal and mean "use that fraction of the shard
list".  When a themed domain is capped, its shortfall is redistributed to
`general`; if `general` also caps out, the residual is logged as an unreachable
part of the token target.

Usage
-----
    python -m src.data.mix
    python -m src.data.mix --config configs/session.toml --total-tokens 6e9
    python -m src.data.mix --no-val            # skip the val carve (re-planning)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from src.config import load_config
from src.data import common as C

VALSPLIT = "_valsplit.json"


# --------------------------------------------------------------------------- #

def scan_domain(domain: str) -> dict:
    """Count docs + chars per shard for one domain."""
    d = C.CLEAN_DIR / domain
    files = []
    docs = chars = 0
    for path in C.shard_paths(d):
        n = c = 0
        for rec in C.read_jsonl(path):
            n += 1
            c += len(rec.get("text", ""))
        files.append({"file": str(path.relative_to(C.REPO_ROOT)), "docs": n,
                      "chars": c, "est_tokens": C.est_tokens(c)})
        docs += n
        chars += c
    val_path = d / f"_val_{domain}.jsonl"
    val = None
    if val_path.exists():
        n = c = 0
        for rec in C.read_jsonl(val_path):
            n += 1
            c += len(rec.get("text", ""))
        val = {"file": str(val_path.relative_to(C.REPO_ROOT)), "docs": n,
               "chars": c, "est_tokens": C.est_tokens(c)}
    return {"domain": domain, "files": files, "docs": docs, "chars": chars,
            "est_tokens": C.est_tokens(chars), "val": val,
            "present": bool(files) or val is not None}


def carve_val(domain: str, val_tokens: int, force: bool) -> bool:
    """Split ``val_tokens`` off the tail of a domain.  Returns True if it carved."""
    d = C.CLEAN_DIR / domain
    marker = d / VALSPLIT
    if marker.exists() and not force:
        return False
    shards = C.shard_paths(d)
    if not shards:
        return False

    total_chars = sum(
        len(rec.get("text", "")) for p in shards for rec in C.read_jsonl(p)
    )
    # Never take more than 40% of a domain for validation (matters for the smoke
    # test and for tiny domains like haiku).
    want_chars = min(val_tokens * C.CHARS_PER_TOKEN, int(total_chars * 0.4))
    if want_chars <= 0:
        return False

    # Walk shards from the end, consuming until we have enough val chars.
    consumed: list[Path] = []
    acc = 0
    for path in reversed(shards):
        consumed.append(path)
        acc += sum(len(rec.get("text", "")) for rec in C.read_jsonl(path))
        if acc >= want_chars:
            break
    consumed.reverse()

    val_path = d / f"_val_{domain}.jsonl"
    rest_path = d / f"{domain}_rest.jsonl"
    # Write to .part first.  On a re-carve the only remaining shard is often
    # `<domain>_rest.jsonl` itself, so opening rest_path for writing directly
    # would truncate the very file we are about to read.
    val_tmp = val_path.with_suffix(".jsonl.part")
    rest_tmp = rest_path.with_suffix(".jsonl.part")
    val_chars = val_docs = rest_docs = rest_chars = 0
    with open(val_tmp, "w", encoding="utf-8") as vf, open(rest_tmp, "w", encoding="utf-8") as rf:
        for path in consumed:
            for rec in C.read_jsonl(path):
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                n = len(rec.get("text", ""))
                if val_chars < want_chars:
                    vf.write(line)
                    val_chars += n
                    val_docs += 1
                else:
                    rf.write(line)
                    rest_chars += n
                    rest_docs += 1
    for path in consumed:
        path.unlink()
    val_tmp.rename(val_path)
    if rest_docs:
        rest_tmp.rename(rest_path)
    else:
        rest_tmp.unlink()

    marker.write_text(json.dumps({
        "val_file": val_path.name, "val_docs": val_docs,
        "val_est_tokens": C.est_tokens(val_chars),
        "rest_file": rest_path.name if rest_docs else None,
        "rest_docs": rest_docs, "rest_est_tokens": C.est_tokens(rest_chars),
        "consumed_shards": [p.name for p in consumed],
    }, indent=2) + "\n")
    print(f"[val] {domain}: {val_docs:,} docs / ~{C.human(C.est_tokens(val_chars))} tokens "
          f"carved from {len(consumed)} shard(s)")
    return True


# --------------------------------------------------------------------------- #

def plan_epochs(scans: dict[str, dict], mix: dict[str, float], total: int, cap: float) -> dict:
    """epochs per domain + shortfall redistribution onto `general`."""
    plan: dict[str, dict] = {}
    shortfall = 0
    for domain, ratio in mix.items():
        avail = scans[domain]["est_tokens"] if scans[domain]["present"] else 0
        target = int(total * ratio)
        if avail == 0:
            plan[domain] = {"target": target, "avail": 0, "epochs": 0.0,
                            "effective": 0, "capped": False, "missing": True}
            shortfall += target
            continue
        epochs = target / avail
        capped = epochs > cap
        if capped:
            epochs = cap
        effective = int(avail * epochs)
        if capped and domain != "general":
            shortfall += target - effective
        plan[domain] = {"target": target, "avail": avail, "epochs": round(epochs, 4),
                        "effective": effective, "capped": capped, "missing": False}

    residual = 0
    if shortfall > 0 and "general" in plan and not plan["general"]["missing"]:
        g = plan["general"]
        g["redistributed"] = shortfall
        new_target = g["target"] + shortfall
        epochs = new_target / g["avail"]
        if epochs > cap:
            residual = new_target - int(g["avail"] * cap)
            epochs = cap
            g["capped"] = True
        g["epochs"] = round(epochs, 4)
        g["effective"] = int(g["avail"] * epochs)
        g["target"] = new_target
    else:
        residual = shortfall
    return {"domains": plan, "shortfall": shortfall, "residual": residual}


def build_manifest(scans: dict[str, dict], plan: dict, cfg, total: int,
                   val_tokens: int, cap: float) -> dict:
    entries = []
    for domain in cfg.data.mix:
        p = plan["domains"][domain]
        if p["missing"]:
            continue
        for f in scans[domain]["files"]:
            entries.append({
                "file": f["file"], "domain": domain, "split": "train",
                "epochs": p["epochs"], "est_tokens": f["est_tokens"], "docs": f["docs"],
            })
    for domain in cfg.data.mix:
        val = scans[domain]["val"]
        if val:
            entries.append({
                "file": val["file"], "domain": domain, "split": "val",
                "epochs": 1.0, "est_tokens": val["est_tokens"], "docs": val["docs"],
            })
    effective_total = sum(plan["domains"][d]["effective"] for d in cfg.data.mix)
    return {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg.train.seed,
        "chars_per_token": C.CHARS_PER_TOKEN,
        "target_total_tokens": total,
        "planned_total_tokens": effective_total,
        "max_domain_epochs": cap,
        "val_tokens_per_domain": val_tokens,
        "mix": dict(cfg.data.mix),
        "domains": {
            d: {
                "epochs": plan["domains"][d]["epochs"],
                "available_tokens": plan["domains"][d]["avail"],
                "effective_tokens": plan["domains"][d]["effective"],
                "target_tokens": plan["domains"][d]["target"],
                "capped": plan["domains"][d]["capped"],
                "docs": scans[d]["docs"],
                "val_tokens": (scans[d]["val"] or {}).get("est_tokens", 0),
                "val_docs": (scans[d]["val"] or {}).get("docs", 0),
                "missing": plan["domains"][d]["missing"],
            }
            for d in cfg.data.mix
        },
        "shortfall_tokens": plan["shortfall"],
        "residual_tokens": plan["residual"],
        "entries": entries,
    }


def write_stats_md(manifest: dict, path: Path) -> None:
    total_eff = max(manifest["planned_total_tokens"], 1)
    lines = [
        "# Corpus mix",
        "",
        f"Generated {manifest['created']} · seed {manifest['seed']} · "
        f"token estimate = chars/{manifest['chars_per_token']}",
        "",
        f"- target total: **{manifest['target_total_tokens']:,}** tokens",
        f"- planned total: **{manifest['planned_total_tokens']:,}** tokens "
        f"({manifest['planned_total_tokens'] / manifest['target_total_tokens']:.1%} of target)",
        f"- max domain epochs: {manifest['max_domain_epochs']}",
        f"- val per domain: {manifest['val_tokens_per_domain']:,} tokens",
        "",
        "## Train",
        "",
        "| domain | shards | docs | avail tokens | epochs | effective tokens | % of mix | target % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    shard_counts = {}
    for e in manifest["entries"]:
        if e["split"] == "train":
            shard_counts[e["domain"]] = shard_counts.get(e["domain"], 0) + 1
    for domain, info in manifest["domains"].items():
        flag = " ⚠︎capped" if info["capped"] else ""
        flag += " ⚠︎missing" if info["missing"] else ""
        lines.append(
            f"| {domain}{flag} | {shard_counts.get(domain, 0)} | {info['docs']:,} | "
            f"{info['available_tokens']:,} | {info['epochs']:.2f} | "
            f"{info['effective_tokens']:,} | "
            f"{info['effective_tokens'] / total_eff:.1%} | "
            f"{manifest['mix'][domain]:.0%} |"
        )
    lines += [
        f"| **total** | {sum(shard_counts.values())} | "
        f"{sum(i['docs'] for i in manifest['domains'].values()):,} | "
        f"{sum(i['available_tokens'] for i in manifest['domains'].values()):,} | | "
        f"{manifest['planned_total_tokens']:,} | 100.0% | 100% |",
        "",
        "## Validation",
        "",
        "| domain | docs | est tokens |",
        "|---|---:|---:|",
    ]
    for domain, info in manifest["domains"].items():
        lines.append(f"| {domain} | {info['val_docs']:,} | {info['val_tokens']:,} |")
    if manifest["shortfall_tokens"]:
        lines += [
            "",
            f"> **Shortfall:** {manifest['shortfall_tokens']:,} tokens could not be "
            "supplied by capped/missing themed domains and were redistributed to "
            "`general`.",
        ]
    if manifest["residual_tokens"]:
        lines += [
            f"> **Residual:** {manifest['residual_tokens']:,} tokens are unreachable "
            "even after redistribution — download more data or lower the total-token "
            "target.",
        ]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--total-tokens", type=float, default=None, help="override train.total_tokens")
    ap.add_argument("--val-tokens", type=int, default=None, help="override val_tokens_per_domain")
    ap.add_argument("--max-domain-epochs", type=float, default=None)
    ap.add_argument("--no-val", action="store_true", help="don't carve val sets, just re-plan")
    ap.add_argument("--force-val", action="store_true", help="re-carve val even if already split")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    total = int(args.total_tokens) if args.total_tokens else cfg.train.total_tokens
    val_tokens = args.val_tokens or cfg.data.val_tokens_per_domain
    cap = args.max_domain_epochs or cfg.data.max_domain_epochs

    ratio_sum = sum(cfg.data.mix.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        print(f"[mix] WARNING: mix ratios sum to {ratio_sum}, not 1.0")

    if not args.no_val:
        for domain in cfg.data.mix:
            carve_val(domain, val_tokens, args.force_val)

    scans = {d: scan_domain(d) for d in cfg.data.mix}
    for d, s in scans.items():
        if not s["present"]:
            print(f"[mix] WARNING: data/clean/{d}/ is empty — domain will be missing from the mix")

    plan = plan_epochs(scans, cfg.data.mix, total, cap)
    manifest = build_manifest(scans, plan, cfg, total, val_tokens, cap)

    C.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    (C.CLEAN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_stats_md(manifest, C.CLEAN_DIR / "stats.md")

    for domain, info in manifest["domains"].items():
        note = ""
        if info["capped"]:
            note = f"  <- CAPPED at {cap} epochs, shortfall redistributed"
        if info["missing"]:
            note = "  <- MISSING"
        print(f"[mix] {domain:11s} avail={C.human(info['available_tokens']):>7s} "
              f"epochs={info['epochs']:<6.2f} effective={C.human(info['effective_tokens']):>7s}{note}")
    print(f"[mix] planned {manifest['planned_total_tokens']:,} / {total:,} tokens")
    if manifest["residual_tokens"]:
        print(f"[mix] residual {manifest['residual_tokens']:,} tokens unreachable")
    print(f"[mix] wrote {C.CLEAN_DIR / 'manifest.json'} and {C.CLEAN_DIR / 'stats.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

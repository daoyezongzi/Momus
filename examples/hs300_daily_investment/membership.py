"""Reconstruct historical CSI 300 membership from frozen official notices.

Run with pandas, xlrd, openpyxl and pypdf installed. Downloads are opt-in;
default parsing uses the saved 2026-09-08 source snapshot. Membership endpoints
are inclusive calendar dates; use the trading calendar when querying a session.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output/hs300_daily_investment_20260908"
SEARCH_URL = "https://www.csindex.com.cn/csindex-home/search/search-content"
DETAIL_URL = "https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementById?id={}"
ANCHOR_URL = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000300cons.xls"
# Explicit reviewed dates. The six regular notices say Friday *after close*;
# the two merger notices defer to the actual delisting dates, NOT publication.
EVENT_DATES = {
    15044: "2023-12-11", 15267: "2024-06-17", 15471: "2024-12-16",
    15546: "2025-03-04", 15690: "2025-06-16", 1006022: "2025-09-05",
    3006000: "2025-12-15", 3006137: "2026-06-15",
}
DELISTING_SOURCES = {
    15546: ("haitong_delisting.html", "https://www.sse.com.cn/disclosure/announcement/listing/stock/c/c_20250226_10773005.shtml"),
    1006022: ("csic_delisting.pdf", "https://static.cninfo.com.cn/finalpage/2025-08-30/1224625027.PDF"),
}


def fetch(url: str, path: Path) -> None:
    with urlopen(url, timeout=30) as response:
        data = response.read()
    if not data:
        raise ValueError(f"Empty source: {url}")
    path.write_bytes(data)


def download_sources(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    params = dict(lang="cn", searchInput="沪深300", pageNum=1, pageSize=50,
                  sortField="date", dateRange="all", contentType="announcement")
    fetch(SEARCH_URL + "?" + urlencode(params), source / "csindex_search_50.json")
    for notice_id in EVENT_DATES:
        path = source / f"notice_{notice_id}.json"
        fetch(DETAIL_URL.format(notice_id), path)
        notice = json.loads(path.read_text(encoding="utf-8"))["data"]
        for number, item in enumerate(notice["enclosureList"]):
            extension = item["fileUrl"].rsplit(".", 1)[-1]
            fetch(item["fileUrl"], source / f"notice_{notice_id}_attachment_{number}.{extension}")
    for name, url in DELISTING_SOURCES.values():
        fetch(url, source / name)
    fetch(ANCHOR_URL, source / "current_000300cons.xls")


def code(value) -> str:
    value = str(value).split(".")[0].zfill(6)
    if not re.fullmatch(r"\d{6}", value):
        raise ValueError(f"Invalid A-share code {value}")
    return value


def instrument(value) -> str:
    value = code(value)
    return ("SH" if value.startswith("6") else "SZ") + value


def read_event(source: Path, notice_id: int) -> dict:
    notice = json.loads((source / f"notice_{notice_id}.json").read_text(encoding="utf-8"))["data"]
    text = html.unescape(re.sub(r"<[^>]+>", "", notice["content"]))
    effective = pd.Timestamp(EVENT_DATES[notice_id])
    if notice_id in DELISTING_SOURCES:
        filename, extra_source = DELISTING_SOURCES[notice_id]
        p = source / filename
        if p.suffix == ".pdf":
            proof = "".join(page.extract_text() for page in PdfReader(p).pages)
        else:
            proof = p.read_text(encoding="utf-8", errors="replace")
        compact = re.sub(r"\s", "", proof)
        expected = f"{effective.year}年{effective.month}月{effective.day}日"
        if expected not in compact or "终止" not in compact:
            raise ValueError(f"Delisting-date evidence missing: {notice_id}")
        date_rule = "effective on actual delisting date (separate official source)"
    else:
        m = re.search(r"(\d{4})年(\d+)月(\d+)日收市后生效", text)
        if not m:
            raise ValueError(f"Explicit after-close effective date missing: {notice_id}")
        after_close = pd.Timestamp("-".join(m.groups()))
        if after_close.weekday() != 4 or effective != after_close + pd.Timedelta(days=3):
            raise ValueError("Reviewed Friday-to-Monday schedule changed")
        extra_source = ""
        date_rule = f"after {after_close.date()} close; first new-membership session {effective.date()}"
    attachment = notice["enclosureList"][0]
    suffix = attachment["fileUrl"].rsplit(".", 1)[-1]
    path = source / f"notice_{notice_id}_attachment_0.{suffix}"
    if suffix == "pdf":
        # These six frozen attachments put the complete CSI 300 table on page 1.
        page = PdfReader(path).pages[0].extract_text()
        section = re.split(r"中证\s*500", page, maxsplit=1)[0]
        pairs = [re.findall(r"(?<!\d)\d{6}(?!\d)", line) for line in section.splitlines()]
        pairs = [pair for pair in pairs if pair]
        if not all(len(pair) == 2 for pair in pairs):
            raise ValueError(f"Ambiguous PDF row layout: {notice_id}")
        expected_count = int(re.search(r"沪深300指数更换(\d+)只", text).group(1))
        if len(pairs) != expected_count:
            raise ValueError(f"PDF row count differs from official notice: {notice_id}")
    else:
        table = pd.read_excel(path, header=None, dtype=str)
        rows = table.loc[table[0].fillna("").str.zfill(6) == "000300"]
        pairs = [[code(row[2]), code(row[4])] for _, row in rows.iterrows()]
        if len(pairs) != 1:
            raise ValueError(f"Expected one CSI 300 merger replacement: {notice_id}")
    remove = [instrument(pair[0]) for pair in pairs]
    add = [instrument(pair[1]) for pair in pairs]
    if len(set(remove)) != len(remove) or len(set(add)) != len(add) or set(add) & set(remove):
        raise ValueError(f"Duplicate/overlapping member changes: {notice_id}")
    return dict(notice_id=notice_id, publish_date=notice["publishDate"],
                effective_date=str(effective.date()), title=notice["title"],
                remove=remove, add=add, date_rule=date_rule,
                notice_url=DETAIL_URL.format(notice_id), attachment_url=attachment["fileUrl"],
                effective_date_source_url=extra_source)


def reconstruct(out: Path, start: str = "2023-09-01") -> dict:
    source = out / "source_probe"
    search = json.loads((source / "csindex_search_50.json").read_text(encoding="utf-8"))
    if min(item["itemDate"] for item in search["data"]) >= start:
        raise ValueError("Search page does not cover reconstruction start")
    matching = {
        int(item["id"]) for item in search["data"]
        if item["itemDate"] >= start and (
            (re.sub(r"<[^>]+>", "", item["headline"]).startswith("关于")
             and "调整" in item["headline"] and "样本" in item["headline"])
            or "定期调整结果" in item["headline"])
    }
    if matching != set(EVENT_DATES):
        raise ValueError(f"Unreviewed archive changes: {matching ^ set(EVENT_DATES)}")
    anchor = pd.read_excel(source / "current_000300cons.xls")
    anchor_dates = anchor.iloc[:, 0].astype(str).unique()
    if len(anchor_dates) != 1:
        raise ValueError("Ambiguous snapshot date")
    anchor_date = pd.to_datetime(anchor_dates[0]).date().isoformat()
    current = {instrument(value) for value in anchor.iloc[:, 4]}
    if len(current) != 300 or len(anchor) != 300:
        raise ValueError("Current official snapshot does not contain 300 unique members")
    events = sorted([read_event(source, i) for i in EVENT_DATES], key=lambda e: e["effective_date"])
    if events[-1]["effective_date"] > anchor_date:
        raise ValueError("Snapshot precedes latest reviewed event")
    snapshots = []
    end = pd.Timestamp(anchor_date)
    for event in reversed(events):
        effective = pd.Timestamp(event["effective_date"])
        adds, removes = set(event["add"]), set(event["remove"])
        if not adds <= current or removes & current:
            raise ValueError(f"Reverse transition incompatible with snapshot: {event['notice_id']}")
        snapshots.append((effective, end, set(current)))
        current = (current - adds) | removes
        if len(current) != 300:
            raise ValueError("Historical snapshot does not contain 300 members")
        end = effective - pd.Timedelta(days=1)
    snapshots.append((pd.Timestamp(start), end, set(current)))
    rows = []
    for name in sorted(set().union(*(members for _, _, members in snapshots))):
        intervals = []
        for begin, finish, members in sorted(snapshots):
            if name not in members:
                continue
            if intervals and intervals[-1][1] + pd.Timedelta(days=1) == begin:
                intervals[-1][1] = finish
            else:
                intervals.append([begin, finish])
        rows.extend(dict(instrument=name, start_date=str(begin.date()), end_date=str(finish.date()))
                    for begin, finish in intervals)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "membership.csv", index=False)
    period_checks = [dict(start_date=str(begin.date()), end_date=str(finish.date()), count=len(members))
                     for begin, finish, members in sorted(snapshots)]
    pd.DataFrame(period_checks).to_csv(out / "membership_period_checks.csv", index=False)
    (out / "membership_events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    evidence = []
    for path in sorted(source.glob("*")):
        if path.is_file():
            evidence.append(dict(path=str(path.relative_to(out)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    status = dict(status="RECONSTRUCTED_OFFICIAL_EVENTS", start_date=start, anchor_date=anchor_date,
                  intervals=len(frame), unique_instruments=frame.instrument.nunique(),
                  event_count=len(events), every_period_has_300=True,
                  membership_endpoints="inclusive calendar dates; select actual trading sessions",
                  pit_date_semantics_verified=True,
                  archive_completeness_independently_verified=False,
                  limitations=["Official search archive plus current snapshot reverse reconstruction; independent vendor membership cross-check remains outstanding.",
                               "First interval is left-censored at requested start; it is not the historical first index entry date.",
                               "Membership is not tradability; merger suspensions and conversion proceeds must be modeled separately."],
                  sources=evidence)
    (out / "membership_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--xlrd-wheel", type=Path)
    args = parser.parse_args()
    if args.xlrd_wheel:
        sys.path.insert(0, str(args.xlrd_wheel.resolve()))
    if args.download:
        download_sources(args.out / "source_probe")
    result = reconstruct(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

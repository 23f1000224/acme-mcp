
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager

import requests
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount


ROOT_URL = os.environ["LEDGER_URL"]
TZ = ZoneInfo("Asia/Kolkata")

mcp = FastMCP(
    "Acme Appliances Ledger",
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
)

_orders = None
_rates = None


def get_json(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def page_url(first_url, page):
    parsed = urlparse(first_url)
    params = parse_qs(parsed.query)
    params["page"] = [str(page)]
    return urlunparse(
        parsed._replace(query=urlencode(params, doseq=True))
    )


def load_data():
    global _orders, _rates

    if _orders is not None:
        return

    discovery = get_json(ROOT_URL)
    _rates = get_json(discovery["links"]["rates"])["usd_per_unit"]

    first_url = discovery["links"]["orders"]
    first_page = get_json(first_url)
    records = list(first_page["orders"])
    pages = int(first_page["pages"])

    urls = [page_url(first_url, p) for p in range(2, pages + 1)]

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(get_json, url) for url in urls]
        for future in as_completed(futures):
            records.extend(future.result()["orders"])

    latest = {}
    for order in records:
        order_id = order["id"]
        updated = datetime.fromisoformat(
            order["updated_at"].replace("Z", "+00:00")
        )
        previous = latest.get(order_id)

        if previous is None:
            latest[order_id] = order
        else:
            previous_updated = datetime.fromisoformat(
                previous["updated_at"].replace("Z", "+00:00")
            )
            if updated > previous_updated:
                latest[order_id] = order

    _orders = list(latest.values())


def order_date(order):
    dt = datetime.fromisoformat(
        order["created_at"].replace("Z", "+00:00")
    )
    return dt.astimezone(TZ)


def amount_usd(order):
    return float(order["amount"]) * float(_rates[order["currency"]])


def filters(question):
    q = question.lower()
    result = []

    for region in ["north", "south", "east", "west", "central"]:
        if re.search(r"\b" + region + r"\b", q):
            result.append(
                lambda o, r=region: o["region"].lower() == r
            )
            break

    products = [
        "Air Fryer", "Rice Cooker", "Mixer", "Juicer",
        "Blender", "Grinder", "Kettle", "Toaster"
    ]
    for product in products:
        if product.lower() in q:
            result.append(
                lambda o, p=product: o["product"].lower() == p.lower()
            )
            break

    customer = re.search(r"\bC\d{4}\b", question, re.I)
    if customer:
        cid = customer.group().upper()
        result.append(
            lambda o, c=cid: o["customer"].upper() == c
        )

    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11,
        "december": 12,
    }

    for name, number in months.items():
        if re.search(r"\b" + name + r"\b", q):
            year_match = re.search(r"\b20\d{2}\b", q)
            year = int(year_match.group()) if year_match else None

            def date_filter(o, n=number, y=year):
                d = order_date(o)
                return d.month == n and (y is None or d.year == y)

            result.append(date_filter)
            break

    return result


def apply_filters(orders, predicates):
    return [o for o in orders if all(p(o) for p in predicates)]


@mcp.tool()
def answer_question(question: str) -> dict:
    """Answer questions about orders, revenue, refunds, average paid
    order value, distinct customers, products and quantities.
    Monetary answers are in USD."""
    try:
        load_data()
        q = question.lower()
        selected = apply_filters(_orders, filters(question))

        if "refund" in q or "refunded" in q:
            refunded = [
                o for o in selected
                if o["status"].lower() == "refunded"
            ]
            if any(w in q for w in ["how many", "number of", "count"]):
                return {"answer": len(refunded)}
            return {
                "answer": round(sum(amount_usd(o) for o in refunded), 2)
            }

        paid = [
            o for o in selected if o["status"].lower() == "paid"
        ]

        if "distinct customer" in q or "unique customer" in q:
            return {"answer": len({o["customer"] for o in paid})}

        if any(w in q for w in ["average", "avg", "mean"]):
            if not paid:
                return {"answer": 0}
            return {
                "answer": round(
                    sum(amount_usd(o) for o in paid) / len(paid), 2
                )
            }

        if any(w in q for w in ["how many", "number of", "count"]):
            if any(w in q for w in ["paid", "bought", "purchased"]):
                return {"answer": len(paid)}
            return {"answer": len(selected)}

        if any(w in q for w in [
            "which product", "top product", "best-selling product",
            "most revenue", "earned the most"
        ]):
            totals = {}
            for o in paid:
                totals[o["product"]] = (
                    totals.get(o["product"], 0) + amount_usd(o)
                )
            if not totals:
                return {"answer": "No matching paid orders"}
            return {"answer": max(totals, key=totals.get)}

        if any(w in q for w in [
            "which customer", "top customer", "best customer"
        ]):
            totals = {}
            for o in paid:
                totals[o["customer"]] = (
                    totals.get(o["customer"], 0) + amount_usd(o)
                )
            if not totals:
                return {"answer": "No matching paid orders"}
            return {"answer": max(totals, key=totals.get)}

        if any(w in q for w in [
            "quantity", "units sold", "how many units"
        ]):
            return {"answer": sum(o["qty"] for o in paid)}

        return {
            "answer": round(sum(amount_usd(o) for o in paid), 2)
        }

    except Exception:
        return {"answer": "Unable to retrieve or process ledger data."}


@asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


app = Starlette(
    routes=[
        Mount("/mcp", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)

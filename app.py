
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
    _rates = get_json(
        discovery["links"]["rates"]
    )["usd_per_unit"]

    first_url = discovery["links"]["orders"]
    first_page = get_json(first_url)

    records = list(first_page["orders"])
    pages = int(first_page["pages"])

    urls = [
        page_url(first_url, page)
        for page in range(2, pages + 1)
    ]

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [
            pool.submit(get_json, url)
            for url in urls
        ]

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
    currency = order["currency"]
    return float(order["amount"]) * float(_rates[currency])


def filters(question):
    q = question.lower()
    result = []

    for region in [
        "north", "south", "east", "west", "central"
    ]:
        if re.search(r"\b" + region + r"\b", q):
            result.append(
                lambda order, r=region:
                order["region"].lower() == r
            )
            break

    products = [
        "Air Fryer",
        "Rice Cooker",
        "Mixer",
        "Juicer",
        "Blender",
        "Grinder",
        "Kettle",
        "Toaster",
    ]

    for product in products:
        if product.lower() in q:
            result.append(
                lambda order, p=product:
                order["product"].lower() == p.lower()
            )
            break

    customer = re.search(r"\bC\d{4}\b", question, re.I)

    if customer:
        customer_id = customer.group().upper()
        result.append(
            lambda order, c=customer_id:
            order["customer"].upper() == c
        )

    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    for name, number in months.items():
        if re.search(r"\b" + name + r"\b", q):
            year_match = re.search(r"\b20\d{2}\b", q)
            year = (
                int(year_match.group())
                if year_match else None
            )

            def date_filter(
                order, n=number, y=year
            ):
                date = order_date(order)
                return (
                    date.month == n
                    and (y is None or date.year == y)
                )

            result.append(date_filter)
            break

    return result


def apply_filters(orders, predicates):
    return [
        order
        for order in orders
        if all(predicate(order) for predicate in predicates)
    ]


@mcp.tool()
def answer_question(question: str) -> dict:
    """Answer questions about orders, revenue, refunds,
    average paid order value, customers, products and quantities.
    Monetary answers are in USD."""

    try:
        load_data()

        q = question.lower()
        selected = apply_filters(
            _orders, filters(question)
        )

        # Refund counts and amounts
        if "refund" in q or "refunded" in q:
            refunded = [
                order for order in selected
                if order["status"].lower() == "refunded"
            ]

            if any(
                word in q
                for word in ["how many", "number of", "count"]
            ):
                return {"answer": len(refunded)}

            return {
                "answer": round(
                    sum(amount_usd(o) for o in refunded),
                    2,
                )
            }

        # Only paid orders count as revenue
        paid = [
            order for order in selected
            if order["status"].lower() == "paid"
        ]

        # Distinct customers
        if (
            "distinct customer" in q
            or "unique customer" in q
        ):
            return {
                "answer": len({
                    order["customer"] for order in paid
                })
            }

        # Average paid order value
        if any(
            word in q for word in ["average", "avg", "mean"]
        ):
            if not paid:
                return {"answer": 0}

            return {
                "answer": round(
                    sum(amount_usd(o) for o in paid)
                    / len(paid),
                    2,
                )
            }

        # Count orders
        if any(
            word in q
            for word in ["how many", "number of", "count"]
        ):
            if any(
                word in q
                for word in ["paid", "bought", "purchased"]
            ):
                return {"answer": len(paid)}

            return {"answer": len(selected)}

        # Product with highest paid revenue
        if any(
            phrase in q
            for phrase in [
                "which product",
                "top product",
                "best-selling product",
                "most revenue",
                "earned the most",
            ]
        ):
            totals = {}

            for order in paid:
                product = order["product"]
                totals[product] = (
                    totals.get(product, 0)
                    + amount_usd(order)
                )

            if not totals:
                return {
                    "answer": "No matching paid orders"
                }

            return {
                "answer": max(totals, key=totals.get)
            }

        # Customer with highest paid revenue
        if any(
            phrase in q
            for phrase in [
                "which customer",
                "top customer",
                "best customer",
            ]
        ):
            totals = {}

            for order in paid:
                customer = order["customer"]
                totals[customer] = (
                    totals.get(customer, 0)
                    + amount_usd(order)
                )

            if not totals:
                return {
                    "answer": "No matching paid orders"
                }

            return {
                "answer": max(totals, key=totals.get)
            }

        # Units sold
        if any(
            phrase in q
            for phrase in [
                "quantity", "units sold", "how many units"
            ]
        ):
            return {
                "answer": sum(order["qty"] for order in paid)
            }

        # Default: total paid revenue in USD
        return {
            "answer": round(
                sum(amount_usd(order) for order in paid),
                2,
            )
        }

    except Exception:
        return {
            "answer": "Unable to retrieve or process ledger data."
        }


# Configure the MCP application to handle requests at its
# mounted root, so the public endpoint is /mcp.
mcp.settings.streamable_http_path = "/"


@asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


app = Starlette(
    routes=[
        Mount(
            "/mcp",
            app=mcp.streamable_http_app(),
        ),
    ],
    lifespan=lifespan,
)

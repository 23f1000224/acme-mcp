
import os
import re
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager

import requests
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def parse_datetime(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt


def load_data():
    global _orders, _rates

    if _orders is not None:
        return

    logger.info("Loading ledger data")
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
        futures = [pool.submit(get_json, url) for url in urls]

        for future in as_completed(futures):
            records.extend(future.result()["orders"])

    # Retain the latest version of every order.
    latest = {}

    for order in records:
        order_id = order["id"]
        updated = parse_datetime(order["updated_at"])
        previous = latest.get(order_id)

        if previous is None:
            latest[order_id] = order
        elif updated > parse_datetime(previous["updated_at"]):
            latest[order_id] = order

    _orders = list(latest.values())
    logger.info("Loaded %d unique orders", len(_orders))


def order_date(order):
    return parse_datetime(order["created_at"]).astimezone(TZ)


def amount_usd(order):
    currency = order["currency"]

    if currency not in _rates:
        raise ValueError(f"Missing USD conversion rate for {currency}")

    return float(order["amount"]) * float(_rates[currency])


def filters(question):
    q = question.lower()
    result = []

    # Region
    for region in ["north", "south", "east", "west", "central"]:
        if re.search(r"\b" + region + r"\b", q):
            result.append(
                lambda o, r=region: o["region"].lower() == r
            )
            break

    # Product
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
                lambda o, p=product: o["product"].lower() == p.lower()
            )
            break

    # Customer ID
    customer = re.search(r"\bC\d{4}\b", question, re.I)

    if customer:
        cid = customer.group().upper()
        result.append(
            lambda o, c=cid: o["customer"].upper() == c
        )

    # Month and optional year
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
            year = int(year_match.group()) if year_match else None

            def date_filter(o, n=number, y=year):
                d = order_date(o)
                return d.month == n and (y is None or d.year == y)

            result.append(date_filter)
            break

    return result


def apply_filters(orders, predicates):
    return [
        order for order in orders
        if all(predicate(order) for predicate in predicates)
    ]


def is_count_question(q):
    return any(
        phrase in q
        for phrase in ["how many", "number of", "count"]
    )


def is_product_revenue_ranking(q):
    normalized = q.replace("-", " ")

    phrases = [
        "top product",
        "top selling product",
        "best selling product",
        "which product",
        "most revenue",
        "earned the most",
        "highest revenue",
        "highest earning product",
        "product by revenue",
        "revenue by product",
        "sold the most",
        "top selling",
        "best selling",
    ]

    return any(phrase in normalized for phrase in phrases)


def is_customer_revenue_ranking(q):
    return any(
        phrase in q
        for phrase in [
            "which customer",
            "top customer",
            "best customer",
            "highest revenue customer",
            "customer who spent the most",
            "customer with the most revenue",
        ]
    )


@mcp.tool()
def answer_question(question: str) -> dict:
    """Answer questions about orders, revenue, refunds,
    average paid order value, distinct customers, products,
    and quantities. Monetary answers are in USD."""

    try:
        load_data()
        q = question.lower()

        selected = apply_filters(_orders, filters(question))

        # Refund questions
        if "refund" in q or "refunded" in q:
            refunded = [
                o for o in selected
                if o["status"].lower() == "refunded"
            ]

            if is_count_question(q):
                return {"answer": len(refunded)}

            return {
                "answer": round(
                    sum(amount_usd(o) for o in refunded), 2
                )
            }

        # Only paid orders count as sales.
        paid = [
            o for o in selected
            if o["status"].lower() == "paid"
        ]

        # Distinct customers who bought products.
        if (
            "distinct customer" in q
            or "unique customer" in q
            or "different customers" in q
        ):
            return {
                "answer": len({o["customer"] for o in paid})
            }

        # Product with the highest total paid revenue.
        if is_product_revenue_ranking(q):
            totals = {}

            for order in paid:
                product = order["product"]
                totals[product] = (
                    totals.get(product, 0.0) + amount_usd(order)
                )

            if not totals:
                return {"answer": "No matching paid orders"}

            return {
                "answer": max(totals, key=totals.get)
            }

        # Customer with the highest total paid revenue.
        if is_customer_revenue_ranking(q):
            totals = {}

            for order in paid:
                customer = order["customer"]
                totals[customer] = (
                    totals.get(customer, 0.0) + amount_usd(order)
                )

            if not totals:
                return {"answer": "No matching paid orders"}

            return {
                "answer": max(totals, key=totals.get)
            }

        # Average paid order value, including region/month filters.
        if any(word in q for word in ["average", "avg", "mean"]):
            if not paid:
                return {"answer": 0}

            average = (
                sum(amount_usd(o) for o in paid) / len(paid)
            )

            return {"answer": round(average, 2)}

        # Units sold.
        if any(
            phrase in q
            for phrase in [
                "quantity",
                "units sold",
                "how many units",
                "number of units",
            ]
        ):
            return {
                "answer": sum(int(o["qty"]) for o in paid)
            }

        # Count orders.
        if is_count_question(q):
            if any(
                word in q
                for word in ["paid", "bought", "purchased"]
            ):
                return {"answer": len(paid)}

            return {"answer": len(selected)}

        # Total paid revenue in USD.
        return {
            "answer": round(
                sum(amount_usd(o) for o in paid), 2
            )
        }

    except Exception:
        logger.exception("Failed to answer question: %s", question)
        return {
            "answer": "Unable to retrieve or process ledger data."
        }


async def plain_answer(request):
    try:
        data = await request.json()

        if not isinstance(data, dict):
            return JSONResponse(
                {"error": "Request body must be a JSON object"},
                status_code=400,
            )

        question = data.get("question")

        if not isinstance(question, str) or not question.strip():
            return JSONResponse(
                {"error": "Expected a non-empty question string"},
                status_code=400,
            )

        result = answer_question(question)
        return JSONResponse(result)

    except Exception:
        logger.exception("HTTP request failed")
        return JSONResponse(
            {"error": "Unable to process request"},
            status_code=500,
        )


@asynccontextmanager
async def lifespan(app):
    async with mcp.session_manager.run():
        yield


mcp_app = mcp.streamable_http_app()

app = Starlette(
    routes=[
        Route("/", plain_answer, methods=["POST"]),
        Route("/mcp", plain_answer, methods=["POST"]),
        Route("/mcp/", plain_answer, methods=["POST"]),
        Mount("/mcp-rpc", app=mcp_app),
    ],
    lifespan=lifespan,
)

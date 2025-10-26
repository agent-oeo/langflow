#!/usr/bin/env python3
"""
MCP Server for Airline Tools using FastMCP with SSE (Server-Sent Events)

This server exposes all airline tools from tau-bench as MCP tools with proper JSON schemas.
Run with: python airline_tools_mcp_fixed.py
MCP URL: http://localhost:8000/sse
Reload endpoint: POST http://localhost:8001/reload
"""

import logging
import json
from typing import Any, Dict, List, Annotated
import threading
import asyncio

from pydantic import BaseModel, Field
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.requests import Request
import uvicorn

from tau_bench.envs.airline.data import load_data
from tau_bench.envs.airline.tools import (
    BookReservation,
    Calculate,
    CancelReservation,
    GetReservationDetails,
    GetUserDetails,
    ListAllAirports,
    SearchDirectFlight,
    SearchOnestopFlight,
    SendCertificate,
    Think,
    TransferToHumanAgents,
    UpdateReservationBaggages,
    UpdateReservationFlights,
    UpdateReservationPassengers,
)
from tau_bench.envs.base import consistent_hash, to_hashable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airline-tools-mcp")

# Create FastMCP server
mcp = FastMCP("Airline Tools")

# Load airline data globally (mutable dict)
logger.info("Loading airline data...")
airline_data = load_data()
logger.info(f"Loaded {len(airline_data.get('flights', {}))} flights, "
            f"{len(airline_data.get('reservations', {}))} reservations, "
            f"{len(airline_data.get('users', {}))} users")


def _format_tool_description(tool_class) -> str:
    """Create a rich tool description using the tool's get_info metadata."""
    info = tool_class.get_info().get("function", {})
    description = info.get("description", "").strip()
    properties = info.get("parameters", {}).get("properties", {})

    lines: List[str] = []
    if description:
        lines.append(description)

    if properties:
        if lines:
            lines.append("")
        lines.append("Parameters:")

        for name, schema in properties.items():
            param_type = schema.get("type")
            details: List[str] = []
            if param_type:
                details.append(param_type)

            if param_type == "array":
                item_schema = schema.get("items", {})
                item_type = item_schema.get("type")
                if item_type:
                    details.append(f"items: {item_type}")

            enum_values = schema.get("enum")
            if enum_values:
                formatted_enum = ", ".join(str(value) for value in enum_values)
                details.append(f"options: {formatted_enum}")

            detail_suffix = f" ({', '.join(details)})" if details else ""
            param_description = schema.get("description", "").strip()
            if param_description:
                lines.append(f"- {name}{detail_suffix}: {param_description}")
            else:
                lines.append(f"- {name}{detail_suffix}")

    return "\n".join(lines)


BOOK_RESERVATION_DESCRIPTION = _format_tool_description(BookReservation)
CALCULATE_DESCRIPTION = _format_tool_description(Calculate)
CANCEL_RESERVATION_DESCRIPTION = _format_tool_description(CancelReservation)
GET_RESERVATION_DETAILS_DESCRIPTION = _format_tool_description(GetReservationDetails)
GET_USER_DETAILS_DESCRIPTION = _format_tool_description(GetUserDetails)
LIST_ALL_AIRPORTS_DESCRIPTION = _format_tool_description(ListAllAirports)
SEARCH_DIRECT_FLIGHT_DESCRIPTION = _format_tool_description(SearchDirectFlight)
SEARCH_ONESTOP_FLIGHT_DESCRIPTION = _format_tool_description(SearchOnestopFlight)
SEND_CERTIFICATE_DESCRIPTION = _format_tool_description(SendCertificate)
THINK_DESCRIPTION = _format_tool_description(Think)
TRANSFER_TO_HUMAN_AGENTS_DESCRIPTION = _format_tool_description(TransferToHumanAgents)
UPDATE_RESERVATION_BAGGAGES_DESCRIPTION = _format_tool_description(UpdateReservationBaggages)
UPDATE_RESERVATION_FLIGHTS_DESCRIPTION = _format_tool_description(UpdateReservationFlights)
UPDATE_RESERVATION_PASSENGERS_DESCRIPTION = _format_tool_description(UpdateReservationPassengers)


def get_database_hash() -> str:
    """Compute the current hash of the in-memory airline database."""
    return consistent_hash(to_hashable(airline_data))
# Define Pydantic models for nested structures to preserve all fields
class FlightInfo(BaseModel):
    flight_number: str = Field(description="Flight number, such as 'HAT001'.")
    date: str = Field(description="The date for the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.")

class PassengerInfo(BaseModel):
    first_name: str = Field(description="The first name of the passenger, such as 'Noah'.")
    last_name: str = Field(description="The last name of the passenger, such as 'Brown'.")
    dob: str = Field(description="The date of birth of the passenger in the format 'YYYY-MM-DD', such as '1990-01-01'.")

class PaymentMethod(BaseModel):
    payment_id: str = Field(description="The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.")
    amount: int = Field(description="The amount to be paid.")


def reload_database():
    """Reload the airline database from source files."""
    global airline_data
    logger.info("Reloading airline database...")
    airline_data = load_data()


async def reload_endpoint(request: Request):
    """REST endpoint to reload the database."""
    reload_database()
    return JSONResponse({
        "status": "success",
        "message": "Database reloaded",
        "stats": {
            "flights": len(airline_data.get('flights', {})),
            "reservations": len(airline_data.get('reservations', {})),
            "users": len(airline_data.get('users', {}))
        }
    })


async def hash_endpoint(request: Request):
    """REST endpoint to get the current database hash."""
    db_hash = get_database_hash()
    logger.info("Computed database hash via /hash endpoint: %s", db_hash)
    return JSONResponse({
        "status": "success",
        "hash": db_hash,
    })


# Define Annotated types with descriptions from tau-bench schemas
UserId = Annotated[str, "The ID of the user to book the reservation, such as 'sara_doe_496'."]
OriginIATA = Annotated[str, "The IATA code for the origin city, such as 'SFO'."]
DestinationIATA = Annotated[str, "The IATA code for the destination city, such as 'JFK'."]
FlightType = Annotated[str, "The flight type: 'one_way' or 'round_trip'."]
Cabin = Annotated[str, "The cabin class: 'basic_economy', 'economy', or 'business'."]
# Use Pydantic models for complex nested structures
FlightsList = Annotated[List[FlightInfo], "An array of flight objects with flight_number and date."]
PassengersList = Annotated[List[PassengerInfo], "An array of passenger objects with first_name, last_name, and dob."]
PaymentMethodsList = Annotated[List[PaymentMethod], "An array of payment method objects with payment_id and amount."]
TotalBaggages = Annotated[int, "The total number of baggage items included in the reservation."]
NonfreeBaggages = Annotated[int, "The number of non-free baggage items included in the reservation."]
Insurance = Annotated[str, "Whether to include travel insurance: 'yes' or 'no'."]
ReservationId = Annotated[str, "The reservation ID."]
FlightDate = Annotated[str, "The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'."]
Expression = Annotated[str, "The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces."]
Thought = Annotated[str, "Internal reasoning or thought process."]
Summary = Annotated[str, "Summary of the conversation for human agents."]
CertificateAmount = Annotated[int, "The amount of the certificate to send to the user."]


@mcp.tool(description="Book a reservation for a user with specified flights, passengers, and payment methods.")
def book_reservation(
    user_id: UserId,
    origin: OriginIATA,
    destination: DestinationIATA,
    flight_type: FlightType,
    cabin: Cabin,
    flights: FlightsList,
    passengers: PassengersList,
    payment_methods: PaymentMethodsList,
    total_baggages: TotalBaggages,
    nonfree_baggages: NonfreeBaggages,
    insurance: Insurance,
) -> str:
    """Book a reservation."""
    # Convert Pydantic models to dicts for tau-bench
    flights_dicts = [f.model_dump() for f in flights]
    passengers_dicts = [p.model_dump() for p in passengers]
    payment_methods_dicts = [pm.model_dump() for pm in payment_methods]

    return BookReservation.invoke(
        airline_data,
        user_id,
        origin,
        destination,
        flight_type,
        cabin,
        flights_dicts,
        passengers_dicts,
        payment_methods_dicts,
        total_baggages,
        nonfree_baggages,
        insurance,
    )


@mcp.tool(description="Calculate the result of a mathematical expression.")
def calculate(expression: Expression) -> str:
    """Calculate the result of a mathematical expression."""
    return Calculate.invoke(airline_data, expression)


@mcp.tool(description="Cancel a reservation by its ID.")
def cancel_reservation(reservation_id: ReservationId) -> str:
    """Cancel a reservation."""
    return CancelReservation.invoke(airline_data, reservation_id)


@mcp.tool(description="Get the details of a reservation by its ID.")
def get_reservation_details(reservation_id: ReservationId) -> str:
    """Get the details of a reservation."""
    return GetReservationDetails.invoke(airline_data, reservation_id)


@mcp.tool(description="Get the details of a user, including their reservations.")
def get_user_details(user_id: UserId) -> str:
    """Get the details of a user, including their reservations."""
    return GetUserDetails.invoke(airline_data, user_id)


@mcp.tool(description="List all available airports with their IATA codes and city names. Returns a JSON object mapping IATA codes to city names (e.g., {'JFK': 'New York', 'LAX': 'Los Angeles'}).")
def list_all_airports() -> str:
    """List all airports."""
    return ListAllAirports.invoke(airline_data)


@mcp.tool(description="Search for direct flights between two cities on a specific date. Use three-letter IATA airport codes.")
def search_direct_flight(
    origin: Annotated[str, "The origin city airport in three letters, such as 'JFK'."],
    destination: Annotated[str, "The destination city airport in three letters, such as 'LAX'."],
    date: FlightDate
) -> str:
    """Search direct flights between two cities on a specific date."""
    return SearchDirectFlight.invoke(airline_data, origin, destination, date)


@mcp.tool(description="Search for one-stop flights between two cities on a specific date. Use three-letter IATA airport codes.")
def search_onestop_flight(
    origin: Annotated[str, "The origin city airport in three letters, such as 'JFK'."],
    destination: Annotated[str, "The destination city airport in three letters, such as 'LAX'."],
    date: FlightDate
) -> str:
    """Search one-stop flights between two cities on a specific date."""
    return SearchOnestopFlight.invoke(airline_data, origin, destination, date)


@mcp.tool(description="Send a certificate (voucher/credit) to a user.")
def send_certificate(user_id: UserId, amount: CertificateAmount) -> str:
    """Send a certificate to a user."""
    return SendCertificate.invoke(airline_data, user_id, amount)


@mcp.tool(description="Think about something (internal reasoning). Use this to plan your approach or reason through a problem before taking action.")
def think(thought: Thought) -> str:
    """Think about something (internal reasoning)."""
    return Think.invoke(airline_data, thought)


@mcp.tool(description="Transfer the conversation to human agents when the request cannot be handled automatically.")
def transfer_to_human_agents(summary: Summary) -> str:
    """Transfer to human agents."""
    return TransferToHumanAgents.invoke(airline_data, summary)


@mcp.tool(description="Update the baggage information of an existing reservation.")
def update_reservation_baggages(
    reservation_id: ReservationId,
    total_baggages: TotalBaggages,
    nonfree_baggages: NonfreeBaggages,
    payment_id: Annotated[str, "The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'."],
) -> str:
    """Update the baggage information of a reservation."""
    return UpdateReservationBaggages.invoke(
        airline_data, reservation_id, total_baggages, nonfree_baggages, payment_id
    )


@mcp.tool(description="Update the flight information of a reservation.")
def update_reservation_flights(
    reservation_id: ReservationId,
    cabin: Cabin,
    flights: FlightsList,
    payment_id: Annotated[str, "The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'."],
) -> str:
    """Update the flight information of a reservation."""
    flights_dicts = [f.model_dump() for f in flights]
    return UpdateReservationFlights.invoke(airline_data, reservation_id, cabin, flights_dicts, payment_id)


@mcp.tool(description="Update the passengers of an existing reservation.")
def update_reservation_passengers(
    reservation_id: ReservationId,
    passengers: PassengersList,
) -> str:
    """Update the passengers of a reservation."""
    passengers_dicts = [p.model_dump() for p in passengers]
    return UpdateReservationPassengers.invoke(airline_data, reservation_id, passengers_dicts)


def run_hash_smoke_test() -> None:
    """CLI helper to verify reload and hash behavior."""
    logger.info("Running database reload/hash smoke test")

    reload_database()
    baseline_hash = get_database_hash()
    logger.info("Baseline database hash: %s", baseline_hash)

    reload_database()
    post_reload_hash = get_database_hash()
    if post_reload_hash != baseline_hash:
        raise RuntimeError(
            f"Reload hash mismatch: {post_reload_hash} (expected {baseline_hash})"
        )
    logger.info("Reload hash matches baseline: %s", post_reload_hash)

    certificate_result = SendCertificate.invoke(airline_data, "mia_li_3668", 42)
    logger.info("send_certificate result: %s", certificate_result.strip())
    if not certificate_result.lower().startswith("certificate"):
        raise RuntimeError(f"send_certificate failed: {certificate_result}")

    flight_entry = airline_data["flights"]["HAT001"]["dates"]["2024-05-16"]
    if flight_entry["status"] != "available":
        raise RuntimeError("Expected HAT001 on 2024-05-16 to be available for booking")
    payment_amount = flight_entry["prices"]["economy"]

    booking_result = BookReservation.invoke(
        airline_data,
        user_id="mia_li_3668",
        origin="PHL",
        destination="LGA",
        flight_type="one_way",
        cabin="economy",
        flights=[{"flight_number": "HAT001", "date": "2024-05-16"}],
        passengers=[
            {"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05"},
        ],
        payment_methods=[{"payment_id": "credit_card_1955700", "amount": payment_amount}],
        total_baggages=0,
        nonfree_baggages=0,
        insurance="no",
    )
    try:
        booking_payload = json.loads(booking_result)
    except json.JSONDecodeError as exc:  # pragma: no cover - sanity guard
        raise RuntimeError(f"Book reservation failed: {booking_result}") from exc

    new_reservation_id = booking_payload["reservation_id"]
    logger.info("Booked reservation %s via smoke test", new_reservation_id)

    mutated_hash = get_database_hash()
    if mutated_hash == baseline_hash:
        raise RuntimeError(
            "Database hash unchanged after mutations; expected difference"
        )
    logger.info("Hash after tool mutations: %s", mutated_hash)

    reload_database()
    final_hash = get_database_hash()
    if final_hash != baseline_hash:
        raise RuntimeError(
            f"Database did not reset cleanly: {final_hash} (expected {baseline_hash})"
        )
    logger.info("Reload restored baseline hash: %s", final_hash)


async def run_reload_server(reload_port: int):
    """Run the reload endpoint on a separate port."""
    reload_app = Starlette(
        routes=[
            Route("/reload", reload_endpoint, methods=["POST", "GET"]),
            Route("/hash", hash_endpoint, methods=["GET"]),
        ]
    )
    config = uvicorn.Config(reload_app, host="0.0.0.0", port=reload_port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def start_server(mcp_port: int, reload_port: int) -> None:
    import asyncio
    import threading

    logger.info("Starting Airline Tools MCP Server with SSE")
    logger.info(f"MCP endpoint: http://localhost:{mcp_port}/sse")
    logger.info(f"Reload endpoint: POST http://localhost:{reload_port}/reload")
    logger.info(f"Hash endpoint: GET http://localhost:{reload_port}/hash")

    # Start the reload server in a separate thread
    def start_reload_server():
        asyncio.run(run_reload_server(reload_port))
    reload_thread = threading.Thread(target=start_reload_server, daemon=True)
    reload_thread.start()
    mcp.run(transport="sse", port=mcp_port)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Airline MCP server controls")
    parser.add_argument(
        "--test-reload-hash",
        action="store_true",
        help="Run a smoke test that exercises reload and hash operations.",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=8000,
        help="Port to run the MCP server on.",
    )
    parser.add_argument(
        "--reload-port",
        type=int,
        default=8001,
        help="Port to run the reload server on.",
    )
    args = parser.parse_args()

    if args.test_reload_hash:
        run_hash_smoke_test()
    else:
        start_server(args.mcp_port, args.reload_port)

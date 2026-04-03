import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# CONFIG
# =========================
YEARLY_TICKET_COUNT = {
    2023: 12000,
    2024: 15500,
    2025: 18200,
}

MONTH_WEIGHTS = {
    1: 1.25,
    2: 1.10,
    3: 1.00,
    4: 0.95,
    5: 0.95,
    6: 0.90,
    7: 0.92,
    8: 0.96,
    9: 1.00,
    10: 1.05,
    11: 1.10,
    12: 1.30,
}

SEVERITY_DISTRIBUTION = {
    "P1": 0.08,
    "P2": 0.37,
    "P3": 0.50,
    "Out of Scope": 0.05,
}

TICKET_TYPE_DISTRIBUTION = {
    "Incident": 0.42,
    "Bug": 0.28,
    "Service Request": 0.20,
    "Change Request": 0.10,
}

TEAM_DISTRIBUTION = {
    "L1 Support": 0.45,
    "L2 Support": 0.30,
    "Product Team": 0.15,
    "Engineering": 0.10,
}

SLA_RULES = {
    "P1": {"response_target_min": 60, "resolution_target_min": 240},
    "P2": {"response_target_min": 240, "resolution_target_min": 1440},
    "P3": {"response_target_min": 240, "resolution_target_min": 4320},
    "Out of Scope": {"response_target_min": None, "resolution_target_min": None},
}

BREACH_REASONS = [
    "Internal delay",
    "Waiting for customer",
    "Dependency on product team",
    "Complex root cause investigation",
    "High ticket volume/backlog",
    "Environment/data issue",
]

CUSTOMERS = [
    "Client A", "Client B", "Client C", "Client D", "Client E",
    "Client F", "Client G", "Client H", "Client I", "Client J"
]

PRODUCT_MODULES = [
    "Order Management",
    "Dispatch Planning",
    "Carrier Integration",
    "Tracking",
    "Billing",
    "Reporting",
    "Master Data",
    "API/EDI",
]

# =========================
# HELPERS
# =========================
def weighted_choice(d: dict):
    keys = list(d.keys())
    probs = list(d.values())
    return np.random.choice(keys, p=probs)

def generate_month_counts(total_tickets: int):
    weights = np.array([MONTH_WEIGHTS[m] for m in range(1, 13)], dtype=float)
    weights = weights / weights.sum()
    raw = np.random.multinomial(total_tickets, weights)
    return {month: int(raw[month - 1]) for month in range(1, 13)}

def random_datetime_in_month(year: int, month: int):
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    start = datetime(year, month, 1)
    seconds = int((next_month - start).total_seconds())
    offset = random.randint(0, max(0, seconds - 1))
    return start + timedelta(seconds=offset)

def business_hour_bias(dt: datetime):
    # Bias some tickets toward working hours
    preferred_hours = [8, 9, 10, 11, 13, 14, 15, 16]
    if random.random() < 0.70:
        hour = random.choice(preferred_hours)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        return dt.replace(hour=hour, minute=minute, second=second)
    return dt

def generate_response_and_resolution(severity: str, ticket_type: str, reopened_count: int, escalated: bool):
    rule = SLA_RULES[severity]
    if rule["response_target_min"] is None:
        return None, None, None, None, None, None

    response_target = rule["response_target_min"]
    resolution_target = rule["resolution_target_min"]

    # Base ratio around target
    if severity == "P1":
        response_ratio = np.random.normal(0.95, 0.45)
        resolution_ratio = np.random.normal(1.05, 0.55)
    elif severity == "P2":
        response_ratio = np.random.normal(0.90, 0.40)
        resolution_ratio = np.random.normal(1.00, 0.45)
    else:
        response_ratio = np.random.normal(0.85, 0.35)
        resolution_ratio = np.random.normal(0.95, 0.40)

    # More complexity
    if ticket_type in ["Bug", "Change Request"]:
        resolution_ratio += 0.15
    if escalated:
        resolution_ratio += 0.25
        response_ratio += 0.05
    if reopened_count > 0:
        resolution_ratio += reopened_count * 0.18

    response_minutes = max(5, int(response_target * response_ratio))
    resolution_minutes = max(response_minutes + 10, int(resolution_target * resolution_ratio))

    response_met = response_minutes <= response_target
    resolution_met = resolution_minutes <= resolution_target

    breach_reason = None
    if (not response_met) or (not resolution_met):
        breach_reason = random.choice(BREACH_REASONS)

    return (
        response_target,
        resolution_target,
        response_minutes,
        resolution_minutes,
        response_met,
        resolution_met,
        breach_reason,
    )

def make_ticket_row(ticket_no: int, year: int, month: int):
    created = business_hour_bias(random_datetime_in_month(year, month))
    severity = weighted_choice(SEVERITY_DISTRIBUTION)
    ticket_type = weighted_choice(TICKET_TYPE_DISTRIBUTION)
    team = weighted_choice(TEAM_DISTRIBUTION)
    customer = random.choice(CUSTOMERS)
    module = random.choice(PRODUCT_MODULES)

    # Escalation more likely for P1/P2
    if severity == "P1":
        escalated = np.random.rand() < 0.70
    elif severity == "P2":
        escalated = np.random.rand() < 0.35
    elif severity == "P3":
        escalated = np.random.rand() < 0.12
    else:
        escalated = np.random.rand() < 0.05

    # Reopen more likely if bug or escalated
    if severity == "Out of Scope":
        reopened_count = 0
    else:
        lam = 0.15
        if ticket_type == "Bug":
            lam += 0.25
        if escalated:
            lam += 0.20
        reopened_count = min(np.random.poisson(lam), 3)

    result = generate_response_and_resolution(severity, ticket_type, reopened_count, escalated)

    if severity == "Out of Scope":
        first_response_at = created + timedelta(minutes=random.randint(30, 600))
        resolved_at = created + timedelta(hours=random.randint(24, 168))
        response_target = None
        resolution_target = None
        response_minutes = int((first_response_at - created).total_seconds() / 60)
        resolution_minutes = int((resolved_at - created).total_seconds() / 60)
        response_met = None
        resolution_met = None
        breach_reason = "Handled outside formal SLA scope"
        sla_status = "Out of Scope"
    else:
        (
            response_target,
            resolution_target,
            response_minutes,
            resolution_minutes,
            response_met,
            resolution_met,
            breach_reason,
        ) = result
        first_response_at = created + timedelta(minutes=response_minutes)
        resolved_at = created + timedelta(minutes=resolution_minutes)

        if response_met and resolution_met:
            sla_status = "Met both"
        elif (not response_met) and (not resolution_met):
            sla_status = "Breached both"
        elif not response_met:
            sla_status = "Breached response"
        else:
            sla_status = "Breached resolution"

    affected_users = (
        random.randint(20, 200) if severity == "P1"
        else random.randint(5, 60) if severity == "P2"
        else random.randint(1, 15)
    )

    business_impact = (
        "Critical" if severity == "P1"
        else "High" if severity == "P2"
        else "Moderate" if severity == "P3"
        else "Low"
    )

    waiting_customer_minutes = 0
    if breach_reason == "Waiting for customer":
        waiting_customer_minutes = random.randint(60, 720)

    row = {
        "ticket_id": f"TK-{year}-{ticket_no:06d}",
        "year": year,
        "month": month,
        "created_at": created,
        "first_response_at": first_response_at,
        "resolved_at": resolved_at,
        "customer_name": customer,
        "product_module": module,
        "ticket_type": ticket_type,
        "severity": severity,
        "assigned_team": team,
        "escalated": "Yes" if escalated else "No",
        "reopened_count": int(reopened_count),
        "affected_users": affected_users,
        "business_impact": business_impact,
        "response_target_min": response_target,
        "resolution_target_min": resolution_target,
        "actual_response_min": response_minutes,
        "actual_resolution_min": resolution_minutes,
        "response_sla_met": response_met,
        "resolution_sla_met": resolution_met,
        "overall_sla_status": sla_status,
        "breach_reason": breach_reason,
        "waiting_customer_min": waiting_customer_minutes,
    }
    return row

def generate_year_data(year: int, total_tickets: int):
    rows = []
    month_counts = generate_month_counts(total_tickets)
    counter = 1
    for month, count in month_counts.items():
        for _ in range(count):
            rows.append(make_ticket_row(counter, year, month))
            counter += 1
    return pd.DataFrame(rows)

def build_summary(df: pd.DataFrame):
    in_scope = df[df["severity"] != "Out of Scope"].copy()

    summary_overall = pd.DataFrame([{
        "total_tickets": len(df),
        "in_scope_tickets": len(in_scope),
        "out_of_scope_tickets": int((df["severity"] == "Out of Scope").sum()),
        "response_sla_compliance_rate": round(in_scope["response_sla_met"].mean() * 100, 2),
        "resolution_sla_compliance_rate": round(in_scope["resolution_sla_met"].mean() * 100, 2),
        "avg_response_min": round(in_scope["actual_response_min"].mean(), 2),
        "avg_resolution_min": round(in_scope["actual_resolution_min"].mean(), 2),
        "escalation_rate": round((df["escalated"] == "Yes").mean() * 100, 2),
        "reopen_rate": round((df["reopened_count"] > 0).mean() * 100, 2),
    }])

    severity_summary = (
        in_scope.groupby("severity")
        .agg(
            tickets=("ticket_id", "count"),
            avg_response_min=("actual_response_min", "mean"),
            avg_resolution_min=("actual_resolution_min", "mean"),
            response_sla_rate=("response_sla_met", "mean"),
            resolution_sla_rate=("resolution_sla_met", "mean"),
            escalation_rate=("escalated", lambda s: (s == "Yes").mean()),
            reopen_rate=("reopened_count", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )
    for col in ["avg_response_min", "avg_resolution_min"]:
        severity_summary[col] = severity_summary[col].round(2)
    for col in ["response_sla_rate", "resolution_sla_rate", "escalation_rate", "reopen_rate"]:
        severity_summary[col] = (severity_summary[col] * 100).round(2)

    monthly_summary = (
        in_scope.groupby(["year", "month"])
        .agg(
            tickets=("ticket_id", "count"),
            avg_response_min=("actual_response_min", "mean"),
            avg_resolution_min=("actual_resolution_min", "mean"),
            response_sla_rate=("response_sla_met", "mean"),
            resolution_sla_rate=("resolution_sla_met", "mean"),
        )
        .reset_index()
    )
    monthly_summary["avg_response_min"] = monthly_summary["avg_response_min"].round(2)
    monthly_summary["avg_resolution_min"] = monthly_summary["avg_resolution_min"].round(2)
    monthly_summary["response_sla_rate"] = (monthly_summary["response_sla_rate"] * 100).round(2)
    monthly_summary["resolution_sla_rate"] = (monthly_summary["resolution_sla_rate"] * 100).round(2)

    breach_summary = (
        in_scope[in_scope["overall_sla_status"] != "Met both"]
        .groupby("breach_reason")
        .agg(tickets=("ticket_id", "count"))
        .reset_index()
        .sort_values("tickets", ascending=False)
    )

    return summary_overall, severity_summary, monthly_summary, breach_summary

def save_year_file(df: pd.DataFrame, year: int):
    summary_overall, severity_summary, monthly_summary, breach_summary = build_summary(df)
    output_file = OUTPUT_DIR / f"sla_actual_performance_{year}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="ticket_raw", index=False)
        summary_overall.to_excel(writer, sheet_name="summary_overall", index=False)
        severity_summary.to_excel(writer, sheet_name="summary_by_severity", index=False)
        monthly_summary.to_excel(writer, sheet_name="summary_by_month", index=False)
        breach_summary.to_excel(writer, sheet_name="breach_reason_summary", index=False)

    print(f"Saved: {output_file}")

def main():
    all_frames = []
    for year, total in YEARLY_TICKET_COUNT.items():
        df = generate_year_data(year, total)
        save_year_file(df, year)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    summary_overall, severity_summary, monthly_summary, breach_summary = build_summary(combined)

    combined_file = OUTPUT_DIR / "sla_actual_performance_2023_2025_combined.xlsx"
    with pd.ExcelWriter(combined_file, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="ticket_raw", index=False)
        summary_overall.to_excel(writer, sheet_name="summary_overall", index=False)
        severity_summary.to_excel(writer, sheet_name="summary_by_severity", index=False)
        monthly_summary.to_excel(writer, sheet_name="summary_by_month", index=False)
        breach_summary.to_excel(writer, sheet_name="breach_reason_summary", index=False)

    print(f"Saved: {combined_file}")

if __name__ == "__main__":
    main()

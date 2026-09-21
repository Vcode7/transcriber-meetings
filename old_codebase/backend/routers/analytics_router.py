"""
Analytics router — public endpoint for demo / product evaluation.

GET /analysis         — all analytics records (newest first)
GET /analysis?summary=true  — aggregated stats only

No authentication is required on this router.
"""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query

from services.analytics import fetch_all_analytics, fetch_analytics_summary

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analytics"])


@router.get("/analysis")
async def get_analytics(
    summary: bool = Query(
        default=False,
        description="If true, return only aggregated summary stats instead of all records.",
    ),
):
    """
    Return stored processing analytics — no authentication required.

    Without ?summary: returns every analytics row (up to 1 000) in a
    structured format grouped by pipeline stage, suitable for building
    dashboards and spotting performance bottlenecks.

    With ?summary=true: returns aggregated KPIs (averages, totals, rates)
    computed across all stored records.
    """
    now_str = datetime.now(timezone.utc).isoformat()

    if summary:
        stats = await fetch_analytics_summary()
        return {
            "generated_at": now_str,
            "summary": stats,
        }

    records = await fetch_all_analytics()
    return {
        "count": len(records),
        "generated_at": now_str,
        "analytics": records,
    }

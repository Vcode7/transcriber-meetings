"""Collections router — organize recordings into named folders."""
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy import text

from database import get_db, from_json, dt_to_str
from routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/collections", tags=["collections"])


# ── Pydantic Models ──────────────────────────────────────────────────────────

class CreateCollectionRequest(BaseModel):
    name: str
    description: Optional[str] = ""

class UpdateCollectionRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class MeetingIdsRequest(BaseModel):
    meeting_ids: List[str]

class ReorderRequest(BaseModel):
    meeting_ids: List[str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/")
async def create_collection(
    body: CreateCollectionRequest,
    user=Depends(get_current_user),
):
    """Create a new collection."""
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name cannot be empty")
    if len(name) > 200:
        raise HTTPException(status_code=400, detail="Collection name too long (max 200 chars)")

    now = dt_to_str(datetime.now(timezone.utc))
    coll_id = str(uuid.uuid4())

    async with get_db() as session:
        # Check uniqueness
        existing = (
            await session.execute(
                text("SELECT id FROM meeting_collections WHERE user_id = :uid AND name = :name"),
                {"uid": user["id"], "name": name},
            )
        ).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="A collection with this name already exists")

        await session.execute(
            text(
                "INSERT INTO meeting_collections (id, user_id, name, description, created_at, updated_at) "
                "VALUES (:id, :uid, :name, :desc, :now, :now)"
            ),
            {"id": coll_id, "uid": user["id"], "name": name, "desc": body.description or "", "now": now},
        )
        await session.commit()

    return {
        "id": coll_id,
        "name": name,
        "description": body.description or "",
        "meeting_count": 0,
        "created_at": now,
        "updated_at": now,
    }


@router.get("/")
async def list_collections(user=Depends(get_current_user)):
    """List all collections for the current user with meeting counts."""
    async with get_db() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT c.id, c.name, c.description, c.created_at, c.updated_at, "
                    "  COALESCE(cnt.n, 0) AS meeting_count "
                    "FROM meeting_collections c "
                    "LEFT JOIN ("
                    "  SELECT collection_id, COUNT(*) AS n "
                    "  FROM meeting_collection_items GROUP BY collection_id"
                    ") cnt ON cnt.collection_id = c.id "
                    "WHERE c.user_id = :uid "
                    "ORDER BY c.updated_at DESC"
                ),
                {"uid": user["id"]},
            )
        ).fetchall()

        return [
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "created_at": r[3],
                "updated_at": r[4],
                "meeting_count": r[5],
            }
            for r in rows
        ]


@router.get("/{collection_id}")
async def get_collection(
    collection_id: str,
    sort: str = "manual",
    user=Depends(get_current_user),
):
    """Get collection detail with its meetings."""
    async with get_db() as session:
        # Fetch collection
        coll = (
            await session.execute(
                text(
                    "SELECT id, name, description, created_at, updated_at "
                    "FROM meeting_collections WHERE id = :cid AND user_id = :uid"
                ),
                {"cid": collection_id, "uid": user["id"]},
            )
        ).fetchone()
        if not coll:
            raise HTTPException(status_code=404, detail="Collection not found")

        # Determine sort order
        if sort == "newest":
            order_clause = "r.created_at DESC"
        elif sort == "oldest":
            order_clause = "r.created_at ASC"
        else:  # manual
            order_clause = "ci.display_order ASC, ci.added_at ASC"

        # Fetch meetings in this collection
        meetings_rows = (
            await session.execute(
                text(
                    f"SELECT r.id, r.filename, r.duration, r.status, "
                    f"r.speakers_detected, r.summary, r.created_at, ci.display_order "
                    f"FROM meeting_collection_items ci "
                    f"JOIN recordings r ON r.id = ci.meeting_id "
                    f"WHERE ci.collection_id = :cid "
                    f"ORDER BY {order_clause}"
                ),
                {"cid": collection_id},
            )
        ).fetchall()

        meetings = []
        for m in meetings_rows:
            meetings.append({
                "id": m[0],
                "filename": m[1],
                "duration": m[2] or 0,
                "status": m[3],
                "speakers_detected": from_json(m[4], []),
                "has_summary": bool(m[5]),
                "created_at": m[6],
                "display_order": m[7],
            })

        return {
            "id": coll[0],
            "name": coll[1],
            "description": coll[2],
            "created_at": coll[3],
            "updated_at": coll[4],
            "meeting_count": len(meetings),
            "meetings": meetings,
        }


@router.patch("/{collection_id}")
async def update_collection(
    collection_id: str,
    body: UpdateCollectionRequest,
    user=Depends(get_current_user),
):
    """Update collection name and/or description."""
    async with get_db() as session:
        # Verify ownership
        coll = (
            await session.execute(
                text("SELECT id FROM meeting_collections WHERE id = :cid AND user_id = :uid"),
                {"cid": collection_id, "uid": user["id"]},
            )
        ).fetchone()
        if not coll:
            raise HTTPException(status_code=404, detail="Collection not found")

        updates = []
        params = {"cid": collection_id, "uid": user["id"]}

        if body.name is not None:
            name = body.name.strip()
            if not name:
                raise HTTPException(status_code=400, detail="Collection name cannot be empty")
            if len(name) > 200:
                raise HTTPException(status_code=400, detail="Collection name too long (max 200 chars)")
            # Check uniqueness
            dup = (
                await session.execute(
                    text(
                        "SELECT id FROM meeting_collections "
                        "WHERE user_id = :uid AND name = :name AND id != :cid"
                    ),
                    {"uid": user["id"], "name": name, "cid": collection_id},
                )
            ).fetchone()
            if dup:
                raise HTTPException(status_code=409, detail="A collection with this name already exists")
            updates.append("name = :name")
            params["name"] = name

        if body.description is not None:
            updates.append("description = :desc")
            params["desc"] = body.description

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        now = dt_to_str(datetime.now(timezone.utc))
        updates.append("updated_at = :now")
        params["now"] = now

        set_clause = ", ".join(updates)
        await session.execute(
            text(f"UPDATE meeting_collections SET {set_clause} WHERE id = :cid AND user_id = :uid"),
            params,
        )
        await session.commit()

        # Return updated collection
        updated = (
            await session.execute(
                text(
                    "SELECT c.id, c.name, c.description, c.created_at, c.updated_at, "
                    "COALESCE((SELECT COUNT(*) FROM meeting_collection_items WHERE collection_id = c.id), 0) AS cnt "
                    "FROM meeting_collections c WHERE c.id = :cid"
                ),
                {"cid": collection_id},
            )
        ).fetchone()

        return {
            "id": updated[0],
            "name": updated[1],
            "description": updated[2],
            "created_at": updated[3],
            "updated_at": updated[4],
            "meeting_count": updated[5],
        }


@router.delete("/{collection_id}")
async def delete_collection(
    collection_id: str,
    user=Depends(get_current_user),
):
    """Delete a collection. Only removes the collection and its links, never the meetings."""
    async with get_db() as session:
        coll = (
            await session.execute(
                text("SELECT id FROM meeting_collections WHERE id = :cid AND user_id = :uid"),
                {"cid": collection_id, "uid": user["id"]},
            )
        ).fetchone()
        if not coll:
            raise HTTPException(status_code=404, detail="Collection not found")

        # Delete junction rows first, then the collection
        await session.execute(
            text("DELETE FROM meeting_collection_items WHERE collection_id = :cid"),
            {"cid": collection_id},
        )
        await session.execute(
            text("DELETE FROM meeting_collections WHERE id = :cid AND user_id = :uid"),
            {"cid": collection_id, "uid": user["id"]},
        )
        await session.commit()

    return {"message": "Collection deleted"}


@router.post("/{collection_id}/meetings")
async def add_meetings(
    collection_id: str,
    body: MeetingIdsRequest,
    user=Depends(get_current_user),
):
    """Add one or more meetings to a collection."""
    if not body.meeting_ids:
        raise HTTPException(status_code=400, detail="No meeting IDs provided")

    async with get_db() as session:
        # Verify collection ownership
        coll = (
            await session.execute(
                text("SELECT id FROM meeting_collections WHERE id = :cid AND user_id = :uid"),
                {"cid": collection_id, "uid": user["id"]},
            )
        ).fetchone()
        if not coll:
            raise HTTPException(status_code=404, detail="Collection not found")

        # Get current max display_order
        max_order = (
            await session.execute(
                text(
                    "SELECT COALESCE(MAX(display_order), -1) FROM meeting_collection_items "
                    "WHERE collection_id = :cid"
                ),
                {"cid": collection_id},
            )
        ).scalar()

        now = dt_to_str(datetime.now(timezone.utc))
        added = 0

        for mid in body.meeting_ids:
            # Verify the meeting belongs to the user
            rec = (
                await session.execute(
                    text("SELECT id FROM recordings WHERE id = :mid AND user_id = :uid"),
                    {"mid": mid, "uid": user["id"]},
                )
            ).fetchone()
            if not rec:
                continue  # skip meetings that don't exist or don't belong to user

            # Check if already in collection (idempotent)
            exists = (
                await session.execute(
                    text(
                        "SELECT id FROM meeting_collection_items "
                        "WHERE collection_id = :cid AND meeting_id = :mid"
                    ),
                    {"cid": collection_id, "mid": mid},
                )
            ).fetchone()
            if exists:
                continue

            max_order += 1
            await session.execute(
                text(
                    "INSERT INTO meeting_collection_items (id, collection_id, meeting_id, display_order, added_at) "
                    "VALUES (:id, :cid, :mid, :order, :now)"
                ),
                {
                    "id": str(uuid.uuid4()),
                    "cid": collection_id,
                    "mid": mid,
                    "order": max_order,
                    "now": now,
                },
            )
            added += 1

        # Update collection's updated_at
        await session.execute(
            text("UPDATE meeting_collections SET updated_at = :now WHERE id = :cid"),
            {"cid": collection_id, "now": now},
        )
        await session.commit()

    return {"message": f"Added {added} meeting(s)", "added_count": added}


@router.delete("/{collection_id}/meetings")
async def remove_meetings(
    collection_id: str,
    body: MeetingIdsRequest,
    user=Depends(get_current_user),
):
    """Remove meetings from a collection. Does NOT delete the meetings themselves."""
    if not body.meeting_ids:
        raise HTTPException(status_code=400, detail="No meeting IDs provided")

    async with get_db() as session:
        # Verify collection ownership
        coll = (
            await session.execute(
                text("SELECT id FROM meeting_collections WHERE id = :cid AND user_id = :uid"),
                {"cid": collection_id, "uid": user["id"]},
            )
        ).fetchone()
        if not coll:
            raise HTTPException(status_code=404, detail="Collection not found")

        removed = 0
        for mid in body.meeting_ids:
            result = await session.execute(
                text(
                    "DELETE FROM meeting_collection_items "
                    "WHERE collection_id = :cid AND meeting_id = :mid"
                ),
                {"cid": collection_id, "mid": mid},
            )
            removed += result.rowcount

        now = dt_to_str(datetime.now(timezone.utc))
        await session.execute(
            text("UPDATE meeting_collections SET updated_at = :now WHERE id = :cid"),
            {"cid": collection_id, "now": now},
        )
        await session.commit()

    return {"message": f"Removed {removed} meeting(s)", "removed_count": removed}


@router.patch("/{collection_id}/reorder")
async def reorder_meetings(
    collection_id: str,
    body: ReorderRequest,
    user=Depends(get_current_user),
):
    """Set the manual display order for meetings in a collection."""
    async with get_db() as session:
        # Verify collection ownership
        coll = (
            await session.execute(
                text("SELECT id FROM meeting_collections WHERE id = :cid AND user_id = :uid"),
                {"cid": collection_id, "uid": user["id"]},
            )
        ).fetchone()
        if not coll:
            raise HTTPException(status_code=404, detail="Collection not found")

        for idx, mid in enumerate(body.meeting_ids):
            await session.execute(
                text(
                    "UPDATE meeting_collection_items SET display_order = :order "
                    "WHERE collection_id = :cid AND meeting_id = :mid"
                ),
                {"order": idx, "cid": collection_id, "mid": mid},
            )

        now = dt_to_str(datetime.now(timezone.utc))
        await session.execute(
            text("UPDATE meeting_collections SET updated_at = :now WHERE id = :cid"),
            {"cid": collection_id, "now": now},
        )
        await session.commit()

    return {"message": "Order updated"}

"""
Test Reasoning Mode for Short ROM Generation and Action Point from Enhanced Point Generation.

Verifies:
1. 10 long points, each 100+ words.
2. Short ROM Generation with Reasoning Mode ON vs OFF.
3. Action Point from Enhanced Point Generation with Reasoning Mode ON vs OFF.
4. Two-call flow when ON (Reasoning call max 2000 tokens -> Generation call with REASONING BLOCK and think=False).
5. Unchanged normal pipeline when OFF (single normal LLM call with think=False).
6. Valid JSON output in both modes.
7. Zero reasoning text leaking into the final output.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from services.ai_provider import get_provider, QwenProvider
from services.rom_service import RomService


# ==============================================================================
# 10 STAGE 2 DISCUSSION POINTS (EACH > 100 WORDS)
# ==============================================================================
TEN_LONG_POINTS = [
    {
        "id": "pt-1",
        "polished_text": (
            "Alex Rivera presented comprehensive telemetry analysis regarding Kafka partition lag in the event streaming "
            "pipeline during peak traffic hours. The cluster currently runs twelve broker nodes handling approximately "
            "45,000 events per second across thirty-two partitions. Recent monitoring graphs indicated intermittent "
            "rebalancing storms whenever consumer group instances dynamically scaled within the Kubernetes cluster. "
            "The engineering team debated whether to tune the max.poll.interval.ms and session.timeout.ms configurations "
            "or introduce static group membership to stabilize pod restarts. While various team members proposed "
            "benchmarking cooperative sticky assignors against the default range assignor, no concrete implementation "
            "roadmap or change ticket was commissioned today, leaving the architecture as an open informational topic "
            "for future quarterly retrospectives and capacity planning reviews."
        ),
        "speaker": "Alex Rivera",
        "speakers": ["Alex Rivera"],
        "action_owner": None,
    },
    {
        "id": "pt-2",
        "polished_text": (
            "The infrastructure team thoroughly reviewed the stability and latency metrics of the primary Redis cache "
            "cluster, which currently hosts session tokens and transient feature flag valuations. Memory utilization "
            "on the master shards has steadily trended toward 78%, causing noticeable latency degradation during cache "
            "invalidation spikes. Following extensive discussion regarding the advantages of Redis 7.2 multi-threaded I/O "
            "and improved replication backlog buffer efficiency, Marcus Vance explicitly committed to executing the live "
            "migration of the production Redis cluster to version 7.2 by Friday, October 24, 2026, at 5:00 PM UTC. Marcus "
            "confirmed that he will coordinate with the platform operations on-call team to execute the rolling failover "
            "across three availability zones while guaranteeing zero customer downtime and validating replica synchronization."
        ),
        "speaker": "Marcus Vance",
        "speakers": ["Marcus Vance"],
        "action_owner": "Marcus Vance",
    },
    {
        "id": "pt-3",
        "polished_text": (
            "Sophia Chen led an architectural evaluation regarding the long-term persistence strategy for semi-structured "
            "application telemetry and audit logs. The team thoroughly compared maintaining a standalone document database "
            "like MongoDB against leveraging native PostgreSQL JSONB columns with generalized inverted indexes (GIN). "
            "Benchmark results showed that while MongoDB offered marginally faster ingest latency for raw document writes, "
            "keeping data in the primary PostgreSQL cluster significantly simplified transactional integrity, cross-table "
            "relational joins, and centralized backup policies via pgBackRest. The architecture committee reached consensus "
            "on adopting PostgreSQL JSONB as the permanent standard for audit log storage, formally closing the exploratory "
            "spike without requiring further vendor evaluation or specialized infrastructure provisioning."
        ),
        "speaker": "Sophia Chen",
        "speakers": ["Sophia Chen"],
        "action_owner": None,
    },
    {
        "id": "pt-4",
        "polished_text": (
            "During the database reliability deep dive, Sarah Jenkins presented metrics illustrating severe connection pool "
            "saturation on the primary Aurora PostgreSQL instance during morning peak spikes. With over eighty backend "
            "microservice pods each maintaining twenty idle connections, the total active connection count exceeded the "
            "database connection limit, triggering connection queuing and transient HTTP 504 gateway timeouts. To eliminate "
            "the connection bottleneck and avoid connection starvation, Sarah Jenkins officially took ownership to implement "
            "and deploy PgBouncer connection pooling middleware in transaction mode by November 15, 2026, ensuring that "
            "maximum backend server connections to PostgreSQL are capped at 150 pooled connections while transparently supporting "
            "microservice autoscaling."
        ),
        "speaker": "Sarah Jenkins",
        "speakers": ["Sarah Jenkins"],
        "action_owner": "Sarah Jenkins",
    },
    {
        "id": "pt-5",
        "polished_text": (
            "David Kim shared an operational status update regarding the newly deployed OpenTelemetry collector daemonsets "
            "running across the staging Kubernetes clusters. Over the past seventy-two hours, the collectors processed approximately "
            "1.2 terabytes of span data, maintaining steady CPU utilization under fifteen percent per node without observable memory "
            "leaks. The team observed that tail-based sampling rules successfully filtered out ninety-five percent of healthy HTTP 200 "
            "traces while preserving one hundred percent of error traces and high-latency outlier requests. The evaluation concluded "
            "that the current collector topology meets all throughput targets for the upcoming general availability release, and the "
            "monitoring team approved keeping the existing configuration active without modifications."
        ),
        "speaker": "David Kim",
        "speakers": ["David Kim"],
        "action_owner": None,
    },
    {
        "id": "pt-6",
        "polished_text": (
            "The security engineering guild discussed ongoing vulnerabilities in the internal API gateway authentication "
            "filter during the quarterly architecture review session. Security audit findings revealed that legacy HMAC token "
            "verification relied on deprecated SHA-1 hashing routines in three auxiliary services. Elena Rostova agreed to lead "
            "the complete refactoring of the shared authentication library to mandate OAuth2 with Proof Key for Code Exchange (PKCE) "
            "and RS256 token validation across all internal services. Elena stated that she will publish the updated SDK package "
            "and distribute comprehensive migration instructions to all squad tech leads before the next security audit milestone, "
            "ensuring all microservices smoothly transition off deprecated hash functions without any production downtime."
        ),
        "speaker": "Elena Rostova",
        "speakers": ["Elena Rostova"],
        "action_owner": "Elena Rostova",
    },
    {
        "id": "pt-7",
        "polished_text": (
            "Rachel Green facilitated an exploratory technical discussion on migrating frontend single-page application assets from our legacy "
            "CloudFront distribution to Cloudflare CDN with distributed edge caching workers. Engineering leads reviewed preliminary performance tests "
            "demonstrating a noticeable twenty-millisecond reduction in first-contentful-paint metrics across European and Asian regional clusters. "
            "However, the infrastructure architecture team highlighted potential budget and compliance implications regarding custom SSL wildcard certificates "
            "and our existing enterprise AWS committed annual spend tiers. Because vendor contract renegotiations with our current primary cloud provider "
            "remain actively underway through the upcoming fiscal quarter, the engineering governance group formally agreed to defer any content delivery "
            "network migrations until contractual obligations and discounted pricing bands are officially finalized by corporate procurement."
        ),
        "speaker": "Rachel Green",
        "speakers": ["Rachel Green"],
        "action_owner": None,
    },
    {
        "id": "pt-8",
        "polished_text": (
            "The platform reliability squad reviewed the recent critical incident retrospective concerning transactional deadlocks within the distributed "
            "job scheduler during high-volume end-of-month batch invoice generation. Root cause investigation confirmed that concurrent worker nodes were "
            "simultaneously acquiring row-level locks on customer account ledgers in non-deterministic alphabetical ordering rather than primary key sequence. "
            "To guarantee deterministic transactional execution and permanently eliminate recurring lock acquisition timeouts, James Wilson officially took "
            "charge of refactoring the job scheduler locking protocol to enforce strict timestamp-based ordering by December 1, 2026. James confirmed that he will "
            "author a dedicated database migration, write unit tests, and conduct comprehensive automated stress tests simulating five hundred concurrent "
            "billing workers against the staging database before rolling out the patch to production clusters."
        ),
        "speaker": "James Wilson",
        "speakers": ["James Wilson"],
        "action_owner": "James Wilson",
    },
    {
        "id": "pt-9",
        "polished_text": (
            "Maya Lin presented findings and recommendations from the comprehensive quarterly cost optimization review across our Google Cloud Platform "
            "development workspaces and sandboxes. Non-production compute instances, idle BigQuery analytical datasets, and unattached persistent SSD disks "
            "accounted for nearly thirty-four percent of total monthly platform infrastructure expenditures. Maya proposed introducing automated cron jobs "
            "that suspend non-essential development compute engines every weekday at 8:00 PM local time and restart them automatically at 7:00 AM on business days. "
            "The management team approved the cost optimization strategy, and Maya Lin committed to configuring, testing, and activating the "
            "automated instance shutdown scripts across all development projects by November 5, 2026, targeting twenty thousand dollars in verifiable "
            "monthly recurring savings across cloud infrastructure budgets."
        ),
        "speaker": "Maya Lin",
        "speakers": ["Maya Lin"],
        "action_owner": "Maya Lin",
    },
    {
        "id": "pt-10",
        "polished_text": (
            "Vikram Patel summarized the comprehensive network topology review regarding our outward-facing reverse proxy clusters and ingress edge controllers. "
            "With the recent formal deprecations of TLS 1.0 and TLS 1.1 across public internet regulatory standards, the network operations team examined cipher "
            "suite usage logs across all edge listeners over the previous ninety days. Analysis confirmed that less than 0.02% of inbound client traffic attempted "
            "cryptographic handshakes using outdated protocols, predominantly originating from legacy automated health-check probes. Although everyone acknowledged "
            "the industry trend toward TLS 1.3 exclusivity, the team reaffirmed that no immediate DNS or reverse proxy configuration changes are necessary at this "
            "juncture, preserving current cipher configurations until next year's annual perimeter security audit."
        ),
        "speaker": "Vikram Patel",
        "speakers": ["Vikram Patel"],
        "action_owner": None,
    },
]


# ==============================================================================
# TEST 1: VERIFY 10 LONG POINTS MEET 100+ WORDS REQUIREMENT
# ==============================================================================
def test_all_10_points_exceed_100_words():
    """Verify all 10 test points contain strictly more than 100 words."""
    assert len(TEN_LONG_POINTS) == 10
    for idx, pt in enumerate(TEN_LONG_POINTS, 1):
        word_count = len(pt["polished_text"].split())
        assert word_count >= 100, f"Point {idx} ({pt['id']}) has only {word_count} words; required >= 100 words."


# ==============================================================================
# TEST 2: SHORT ROM GENERATION - REASONING MODE OFF (NORMAL PIPELINE UNCHANGED)
# ==============================================================================
def test_short_rom_generation_reasoning_mode_off():
    """Verify Short ROM Generation with Reasoning Mode OFF.
    - Exactly 1 normal LLM call per agenda.
    - think=False is passed (without thinking mode).
    - Output is a valid JSON array of point strings.
    - Final ROM structure and metadata are properly preserved.
    """
    rom_service = RomService()
    final_rom = {
        "date": "2026-09-21",
        "time": "10:00",
        "agendas": [
            {
                "agenda_id": "A1",
                "title": "Engineering Architecture & Operations",
                "discussion_points": TEN_LONG_POINTS,
            }
        ],
    }

    mock_calls = []

    def mock_infer(prompt, max_new_tokens=4096, task_key=None, think=None):
        mock_calls.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "task_key": task_key,
            "think": think,
        })
        # Simulate normal single-call condensed JSON array response
        return json.dumps([
            "Marcus Vance committed to migrating the production Redis cache cluster to version 7.2 by October 24, 2026, with platform on-call support.",
            "Sarah Jenkins will implement and deploy PgBouncer connection pooling middleware in transaction mode by November 15, 2026, capping DB connections at 150.",
            "Elena Rostova will refactor the internal auth library to enforce OAuth2 PKCE and RS256 token validation across services.",
            "James Wilson took ownership to refactor the job scheduler locking protocol to enforce strict timestamp ordering by December 1, 2026.",
            "Maya Lin will configure automated development instance shutdown scripts by November 5, 2026."
        ])

    with patch("services.ai_provider.get_provider") as mock_get_provider:
        mock_provider = MagicMock()
        del mock_provider.query  # Ensure _infer path is used
        mock_provider._infer = mock_infer
        mock_get_provider.return_value = mock_provider

        result = rom_service.generate_rom_version(final_rom, version="short", reasoning_mode=False)

    # 1. Verify exactly ONE LLM call was executed (single normal LLM call)
    assert len(mock_calls) == 1, f"Expected 1 LLM call when Reasoning Mode is OFF, got {len(mock_calls)}"
    call = mock_calls[0]
    assert call["task_key"] == "rom_version_short"
    assert call["think"] is False, "Reasoning Mode OFF must pass think=False"

    # 2. Verify output is valid condensed points
    assert "agendas" in result
    agenda = result["agendas"][0]
    points = agenda["discussion_points"]
    assert len(points) == 5
    for p in points:
        assert isinstance(p["text"], str)
        assert len(p["text"]) > 0
        # Verify no reasoning markers leaked
        assert "REASONING BLOCK" not in p["text"]
        assert "ANALYSIS INSTRUCTIONS" not in p["text"]


# ==============================================================================
# TEST 3: SHORT ROM GENERATION - REASONING MODE ON (2-CALL FLOW)
# ==============================================================================
def test_short_rom_generation_reasoning_mode_on():
    """Verify Short ROM Generation with Reasoning Mode ON.
    - Exactly 2 LLM calls per agenda:
      1. Reasoning call: max 2000 tokens, requests analysis only, no final response.
      2. Generation call: passes REASONING BLOCK, think=False, produces valid JSON array.
    - Valid JSON array is parsed.
    - Zero reasoning text leaks into the final output.
    """
    rom_service = RomService()
    final_rom = {
        "date": "2026-09-21",
        "time": "10:00",
        "agendas": [
            {
                "agenda_id": "A1",
                "title": "Engineering Architecture & Operations",
                "discussion_points": TEN_LONG_POINTS,
            }
        ],
    }

    mock_calls = []

    def mock_infer(prompt, max_new_tokens=4096, task_key=None, think=None):
        mock_calls.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "task_key": task_key,
            "think": think,
        })
        if task_key == "rom_version_short_reasoning":
            # Call 1: Return detailed analytical reasoning (NOT JSON)
            return (
                "Discussion Analysis & Synthesis Plan:\n"
                "1. Topics pt-1 (Kafka rebalancing), pt-3 (Postgres JSONB audit log), pt-5 (OpenTelemetry telemetry), "
                "pt-7 (Cloudflare CDN evaluation), and pt-10 (TLS cipher suites) are informational updates or closed discussions "
                "with no active work assigned.\n"
                "2. Actionable items identified:\n"
                "   - pt-2: Marcus Vance committed to Redis 7.2 upgrade by Oct 24, 2026.\n"
                "   - pt-4: Sarah Jenkins committed to PgBouncer deployment by Nov 15, 2026.\n"
                "   - pt-6: Elena Rostova committed to OAuth2 PKCE auth refactoring.\n"
                "   - pt-8: James Wilson committed to job scheduler locking refactor by Dec 1, 2026.\n"
                "   - pt-9: Maya Lin committed to dev VM shutdown automation by Nov 5, 2026.\n"
                "3. Plan: Synthesize the 10 points into 5 high-impact condensed discussion points focusing strictly on "
                "action points, decisions, and deadlines."
            )
        else:
            # Call 2: Return valid JSON array
            return json.dumps([
                "Marcus Vance committed to migrating the production Redis cache cluster to version 7.2 by Friday, October 24, 2026, at 5:00 PM UTC.",
                "Sarah Jenkins will deploy PgBouncer connection pooling middleware in transaction mode by November 15, 2026, capping PostgreSQL connections at 150.",
                "Elena Rostova will refactor the internal authentication library to mandate OAuth2 PKCE and RS256 token validation across services.",
                "James Wilson will refactor the job scheduler locking protocol to enforce timestamp-based ordering by December 1, 2026, avoiding deadlocks.",
                "Maya Lin will configure automated development compute engine shutdown scripts by November 5, 2026."
            ])

    with patch("services.ai_provider.get_provider") as mock_get_provider:
        mock_provider = MagicMock()
        del mock_provider.query
        mock_provider._infer = mock_infer
        mock_get_provider.return_value = mock_provider

        result = rom_service.generate_rom_version(final_rom, version="short", reasoning_mode=True)

    # 1. Verify exactly TWO LLM calls were executed in sequence
    assert len(mock_calls) == 2, f"Expected 2 LLM calls when Reasoning Mode is ON, got {len(mock_calls)}"

    call_1 = mock_calls[0]
    call_2 = mock_calls[1]

    # 2. Verify Call 1 (Reasoning call):
    assert call_1["task_key"] == "rom_version_short_reasoning"
    assert call_1["max_new_tokens"] == 2000, f"Call 1 max_new_tokens must be 2000, got {call_1['max_new_tokens']}"
    assert "Analyze and reason about the following agenda discussion" in call_1["prompt"]
    assert "Do NOT generate the final JSON array or finalized response" in call_1["prompt"]

    # 3. Verify Call 2 (Generation call):
    assert call_2["task_key"] == "rom_version_short"
    assert call_2["think"] is False, "Call 2 must have think=False"
    assert "==================================================" in call_2["prompt"]
    assert "REASONING BLOCK:" in call_2["prompt"]
    assert "Discussion Analysis & Synthesis Plan:" in call_2["prompt"]
    assert "Return ONLY a valid JSON array of condensed point objects" in call_2["prompt"]

    # 4. Verify valid output and NO reasoning text leaked
    points = result["agendas"][0]["discussion_points"]
    assert len(points) == 5
    for p in points:
        assert isinstance(p["text"], str)
        assert "REASONING BLOCK" not in p["text"]
        assert "Discussion Analysis & Synthesis Plan" not in p["text"]
        assert "CRITICAL CONSTRAINT" not in p["text"]


# ==============================================================================
# TEST 4: ACTION POINT GENERATION - REASONING MODE OFF (NORMAL PIPELINE UNCHANGED)
# ==============================================================================
def test_action_point_generation_reasoning_mode_off():
    """Verify Action Point from Enhanced Point Generation with Reasoning Mode OFF.
    - Exactly 1 normal LLM call per chunk.
    - think=False is passed (normal LLM call without thinking mode).
    - Output is a valid list of action items parsed from JSON.
    """
    provider = get_provider()

    mock_calls = []

    def mock_infer(prompt, max_new_tokens=4096, task_key=None, think=None):
        mock_calls.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "task_key": task_key,
            "think": think,
        })
        return json.dumps({
            "action_items": [
                {
                    "source_point_id": "pt-2",
                    "task": "Marcus Vance will migrate production Redis cluster to version 7.2 by Friday, October 24, 2026, at 5:00 PM UTC.",
                    "owner": "Marcus Vance",
                    "deadline": "October 24, 2026",
                    "expected_outcome": "Zero downtime rolling failover across three AZs completed."
                },
                {
                    "source_point_id": "pt-4",
                    "task": "Sarah Jenkins to deploy PgBouncer connection pooling in transaction mode by November 15, 2026.",
                    "owner": "Sarah Jenkins",
                    "deadline": "November 15, 2026",
                    "expected_outcome": "Backend connections capped at 150."
                },
                {
                    "source_point_id": "pt-6",
                    "task": "Elena Rostova to refactor shared authentication library to enforce OAuth2 PKCE.",
                    "owner": "Elena Rostova",
                    "deadline": None,
                    "expected_outcome": "Updated SDK package distributed to all tech leads."
                },
                {
                    "source_point_id": "pt-8",
                    "task": "James Wilson to refactor job scheduler locking protocol by December 1, 2026.",
                    "owner": "James Wilson",
                    "deadline": "December 1, 2026",
                    "expected_outcome": "Deterministic timestamp ordering enforced with no deadlocks."
                },
                {
                    "source_point_id": "pt-9",
                    "task": "Maya Lin to configure automated development instance shutdown scripts by November 5, 2026.",
                    "owner": "Maya Lin",
                    "deadline": "November 5, 2026",
                    "expected_outcome": "$20,000 monthly cost reduction."
                },
            ]
        })

    with patch.object(provider, "_infer", side_effect=mock_infer):
        actions = provider.extract_actions_from_enhanced_points(
            TEN_LONG_POINTS,
            chunk_size=10,
            reasoning_mode=False,
        )

    # 1. Verify exactly ONE LLM call was executed for the 10-point chunk
    assert len(mock_calls) == 1, f"Expected 1 LLM call when Reasoning Mode is OFF, got {len(mock_calls)}"
    call = mock_calls[0]
    assert call["task_key"] == "mom_extract_actions"
    assert call["think"] is False, "Reasoning Mode OFF must pass think=False"

    # 2. Verify valid action items were extracted
    assert len(actions) == 5
    owners = [a["owner"] for a in actions]
    assert "Marcus Vance" in owners
    assert "Sarah Jenkins" in owners
    assert "Elena Rostova" in owners
    assert "James Wilson" in owners
    assert "Maya Lin" in owners

    # 3. Verify no reasoning text in fields
    for a in actions:
        assert "REASONING BLOCK" not in a["task"]


# ==============================================================================
# TEST 5: ACTION POINT GENERATION - REASONING MODE ON (2-CALL FLOW)
# ==============================================================================
def test_action_point_generation_reasoning_mode_on():
    """Verify Action Point from Enhanced Point Generation with Reasoning Mode ON.
    - Exactly 2 LLM calls per chunk:
      1. Reasoning call: max 2000 tokens, requests analysis only, no final response.
      2. Generation call: passes REASONING BLOCK, think=False, produces valid JSON.
    - Valid action items extracted.
    - Zero reasoning text leaks into tasks, owners, or deadlines.
    """
    provider = get_provider()

    mock_calls = []

    def mock_infer(prompt, max_new_tokens=4096, task_key=None, think=None):
        mock_calls.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "task_key": task_key,
            "think": think,
        })
        if task_key == "mom_extract_actions_reasoning":
            # Call 1: Detailed actionability and ownership reasoning
            return (
                "Actionability and Ownership Analysis for Points pt-1 to pt-10:\n"
                "- pt-1: Alex Rivera presented Kafka metrics. No commitment or action made. (No action item)\n"
                "- pt-2: Marcus Vance explicitly committed to migrating Redis 7.2 by Oct 24, 2026. (Valid action)\n"
                "- pt-3: Sophia Chen presented PostgreSQL audit log architecture. Committee decided standard, no further work needed. (No action)\n"
                "- pt-4: Sarah Jenkins committed to deploying PgBouncer pooling by Nov 15, 2026. (Valid action)\n"
                "- pt-5: David Kim presented OpenTelemetry status. No action commissioned. (No action)\n"
                "- pt-6: Elena Rostova agreed to refactor auth library to OAuth2 PKCE. (Valid action)\n"
                "- pt-7: Rachel Green discussed Cloudflare CDN. Explicitly deferred. (No action)\n"
                "- pt-8: James Wilson took ownership to refactor job scheduler locking by Dec 1, 2026. (Valid action)\n"
                "- pt-9: Maya Lin committed to configuring dev VM shutdown automation by Nov 5, 2026. (Valid action)\n"
                "- pt-10: Vikram Patel discussed TLS cipher suites. No change needed. (No action)\n"
            )
        else:
            # Call 2: Final JSON generation
            return json.dumps({
                "action_items": [
                    {
                        "source_point_id": "pt-2",
                        "task": "Marcus Vance will execute the live migration of the production Redis cluster to version 7.2 by Friday, October 24, 2026, at 5:00 PM UTC.",
                        "owner": "Marcus Vance",
                        "deadline": "October 24, 2026",
                        "expected_outcome": "Zero customer downtime rolling failover across three availability zones."
                    },
                    {
                        "source_point_id": "pt-4",
                        "task": "Sarah Jenkins will implement and deploy PgBouncer connection pooling middleware in transaction mode by November 15, 2026.",
                        "owner": "Sarah Jenkins",
                        "deadline": "November 15, 2026",
                        "expected_outcome": "Backend connections capped at 150 pooled connections."
                    },
                    {
                        "source_point_id": "pt-6",
                        "task": "Elena Rostova will lead the refactoring of the shared authentication library to mandate OAuth2 PKCE and RS256 token validation.",
                        "owner": "Elena Rostova",
                        "deadline": None,
                        "expected_outcome": "Updated SDK package and migration instructions distributed to all squad tech leads."
                    },
                    {
                        "source_point_id": "pt-8",
                        "task": "James Wilson will refactor the job scheduler locking protocol to enforce timestamp-based ordering by December 1, 2026.",
                        "owner": "James Wilson",
                        "deadline": "December 1, 2026",
                        "expected_outcome": "Deterministic execution eliminating distributed deadlocks."
                    },
                    {
                        "source_point_id": "pt-9",
                        "task": "Maya Lin will configure and activate automated development compute engine shutdown scripts by November 5, 2026.",
                        "owner": "Maya Lin",
                        "deadline": "November 5, 2026",
                        "expected_outcome": "$20,000 in monthly platform cost savings."
                    },
                ]
            })

    with patch.object(provider, "_infer", side_effect=mock_infer):
        actions = provider.extract_actions_from_enhanced_points(
            TEN_LONG_POINTS,
            chunk_size=10,
            reasoning_mode=True,
        )

    # 1. Verify exactly TWO LLM calls were executed in sequence
    assert len(mock_calls) == 2, f"Expected 2 LLM calls when Reasoning Mode is ON, got {len(mock_calls)}"

    call_1 = mock_calls[0]
    call_2 = mock_calls[1]

    # 2. Verify Call 1 (Reasoning call):
    assert call_1["task_key"] == "mom_extract_actions_reasoning"
    assert call_1["max_new_tokens"] == 2000, f"Call 1 max_new_tokens must be 2000, got {call_1['max_new_tokens']}"
    assert "Analyze and reason about the following enhanced meeting discussion points" in call_1["prompt"]
    assert "Provide your analysis and reasoning ONLY. Do NOT generate the final JSON response" in call_1["prompt"]

    # 3. Verify Call 2 (Generation call):
    assert call_2["task_key"] == "mom_extract_actions"
    assert call_2["think"] is False, "Call 2 must have think=False"
    assert "==================================================" in call_2["prompt"]
    assert "REASONING BLOCK:" in call_2["prompt"]
    assert "Actionability and Ownership Analysis for Points pt-1 to pt-10:" in call_2["prompt"]
    assert "Generate ONLY the final output in the existing valid JSON format" in call_2["prompt"]

    # 4. Verify valid output and NO reasoning text leaked
    assert len(actions) == 5
    for a in actions:
        assert isinstance(a["task"], str)
        assert "REASONING BLOCK" not in a["task"]
        assert "Actionability and Ownership Analysis" not in a["task"]
        assert "CRITICAL INSTRUCTIONS" not in a["task"]
        assert a["owner"] in ["Marcus Vance", "Sarah Jenkins", "Elena Rostova", "James Wilson", "Maya Lin"]


# ==============================================================================
# TEST 6: DIRECT COMPARISON (ON vs OFF) FOR BOTH PIPELINES
# ==============================================================================
def test_compare_on_vs_off_both_pipelines():
    """Directly compare Reasoning Mode ON vs OFF across both main pipeline functions."""
    rom_service = RomService()
    provider = get_provider()

    final_rom = {
        "agendas": [
            {
                "agenda_id": "A1",
                "title": "Engineering Architecture",
                "discussion_points": TEN_LONG_POINTS,
            }
        ]
    }

    # Tracking calls for Short ROM
    short_rom_off_calls = []
    short_rom_on_calls = []

    def mock_short_rom_off(prompt, max_new_tokens=4096, task_key=None, think=None):
        short_rom_off_calls.append(task_key)
        return json.dumps(["Point A: Redis upgrade.", "Point B: PgBouncer deployment."])

    def mock_short_rom_on(prompt, max_new_tokens=4096, task_key=None, think=None):
        short_rom_on_calls.append(task_key)
        if task_key == "rom_version_short_reasoning":
            return "Reasoning: Focus on points with firm dates."
        return json.dumps(["Point A: Redis upgrade by Oct 24.", "Point B: PgBouncer deployment by Nov 15."])

    with patch("services.ai_provider.get_provider") as mock_gp:
        mock_p = MagicMock()
        del mock_p.query
        mock_p._infer = mock_short_rom_off
        mock_gp.return_value = mock_p
        res_off = rom_service.generate_rom_version(final_rom, version="short", reasoning_mode=False)

    with patch("services.ai_provider.get_provider") as mock_gp:
        mock_p = MagicMock()
        del mock_p.query
        mock_p._infer = mock_short_rom_on
        mock_gp.return_value = mock_p
        res_on = rom_service.generate_rom_version(final_rom, version="short", reasoning_mode=True)

    # Compare Short ROM call counts
    assert len(short_rom_off_calls) == 1, "Short ROM OFF must be exactly 1 call"
    assert len(short_rom_on_calls) == 2, "Short ROM ON must be exactly 2 calls"
    assert short_rom_on_calls == ["rom_version_short_reasoning", "rom_version_short"]

    # Tracking calls for Action Point Extraction
    actions_off_calls = []
    actions_on_calls = []

    def mock_actions_off(prompt, max_new_tokens=4096, task_key=None, think=None):
        actions_off_calls.append(task_key)
        return json.dumps({"action_items": [{"source_point_id": "pt-2", "task": "Migrate Redis.", "owner": "Marcus", "deadline": None, "expected_outcome": None}]})

    def mock_actions_on(prompt, max_new_tokens=4096, task_key=None, think=None):
        actions_on_calls.append(task_key)
        if task_key == "mom_extract_actions_reasoning":
            return "Reasoning: pt-2 has owner Marcus."
        return json.dumps({"action_items": [{"source_point_id": "pt-2", "task": "Marcus Vance to migrate Redis.", "owner": "Marcus Vance", "deadline": "2026-10-24", "expected_outcome": "Migrated."}]})

    with patch.object(provider, "_infer", side_effect=mock_actions_off):
        act_off = provider.extract_actions_from_enhanced_points(TEN_LONG_POINTS, chunk_size=10, reasoning_mode=False)

    with patch.object(provider, "_infer", side_effect=mock_actions_on):
        act_on = provider.extract_actions_from_enhanced_points(TEN_LONG_POINTS, chunk_size=10, reasoning_mode=True)

    # Compare Action Extraction call counts
    assert len(actions_off_calls) == 1, "Action extraction OFF must be exactly 1 call"
    assert len(actions_on_calls) == 2, "Action extraction ON must be exactly 2 calls"
    assert actions_on_calls == ["mom_extract_actions_reasoning", "mom_extract_actions"]

    # Both produced valid structured output
    assert len(res_off["agendas"][0]["discussion_points"]) == 2
    assert len(res_on["agendas"][0]["discussion_points"]) == 2
    assert len(act_off) == 1
    assert len(act_on) == 1

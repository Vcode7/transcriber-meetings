import json
import pytest
from services.ai_provider import get_provider

# ==============================================================================
# 10 STAGE 2 DISCUSSION POINTS (EACH > 100 WORDS)
# Realistic software engineering / infrastructure technical meeting.
# ==============================================================================
STAGE_2_POINTS = [
    {
        "id": "pt-1",
        "original_point_ids": ["w1-p1"],
        "polished_text": (
            "Alex Rivera presented telemetry analysis regarding Kafka partition lag in the event streaming "
            "pipeline during peak traffic hours. The cluster currently runs twelve broker nodes handling "
            "approximately 45,000 events per second across thirty-two partitions. Recent monitoring graphs "
            "indicated intermittent rebalancing storms whenever consumer group instances dynamically scaled "
            "within the Kubernetes cluster. The engineering team debated whether to tune the max.poll.interval.ms "
            "and session.timeout.ms configurations or introduce static group membership to stabilize pod restarts. "
            "While various team members proposed benchmarking cooperative sticky assignors against the default range "
            "assignor, no concrete implementation roadmap or change ticket was commissioned today, leaving the "
            "architecture as an open informational topic for future quarterly retrospectives."
        ),
        "speaker": "Alex Rivera",
        "speakers": ["Alex Rivera"],
        "action_owner": None,
        "action_items": [],
        "decisions": [],
        "technical_terms": ["Kafka", "partition lag", "rebalancing storms", "max.poll.interval.ms", "Kubernetes"],
        "dates": [],
        "numbers": ["12 broker nodes", "45,000 events/sec", "32 partitions"],
        "timeline_start": 0.0,
        "timeline_end": 180.0,
    },
    {
        "id": "pt-2",
        "original_point_ids": ["w1-p2"],
        "polished_text": (
            "The infrastructure team reviewed the stability of the primary Redis cache cluster, which currently "
            "hosts session tokens and transient feature flag valuations. Memory utilization on the master shards "
            "has steadily trended toward 78%, causing latency degradation during cache invalidation spikes. "
            "Following extensive discussion regarding the advantages of Redis 7.2 multi-threaded I/O and improved "
            "replication backlog buffer efficiency, Marcus Vance explicitly committed to executing the live migration "
            "of the production Redis cluster to version 7.2 by Friday, October 24, 2026, at 5:00 PM UTC. Marcus "
            "confirmed that he will coordinate with the platform operations on-call team to execute the rolling "
            "failover across three availability zones while keeping zero customer downtime and validating replica sync."
        ),
        "speaker": "Marcus Vance",
        "speakers": ["Marcus Vance"],
        "action_owner": "Marcus Vance",
        "action_items": [],
        "decisions": ["Approve Redis 7.2 migration"],
        "technical_terms": ["Redis 7.2", "replication backlog", "rolling failover", "availability zones"],
        "dates": ["October 24, 2026"],
        "numbers": ["78% memory", "3 availability zones", "5:00 PM UTC"],
        "timeline_start": 185.0,
        "timeline_end": 370.0,
    },
    {
        "id": "pt-3",
        "original_point_ids": ["w2-p1"],
        "polished_text": (
            "Sophia Chen led an architectural evaluation regarding the persistence strategy for semi-structured "
            "application telemetry and audit logs. The team thoroughly compared maintaining a standalone document "
            "database like MongoDB against leveraging native PostgreSQL JSONB columns with generalized inverted indexes "
            "(GIN). Benchmark results showed that while MongoDB offered marginally faster ingest latency for raw document "
            "writes, keeping data in the primary PostgreSQL cluster significantly simplified transactional integrity, "
            "cross-table relational joins, and centralized backup policies via pgBackRest. The architecture committee "
            "reached consensus on adopting PostgreSQL JSONB as the permanent standard for audit log storage, formally "
            "closing the exploratory spike without requiring further vendor evaluation or specialized infrastructure "
            "provisioning."
        ),
        "speaker": "Sophia Chen",
        "speakers": ["Sophia Chen"],
        "action_owner": None,
        "action_items": [],
        "decisions": ["Adopt PostgreSQL JSONB for audit logs"],
        "technical_terms": ["PostgreSQL", "JSONB", "GIN index", "MongoDB", "pgBackRest"],
        "dates": [],
        "numbers": [],
        "timeline_start": 375.0,
        "timeline_end": 540.0,
    },
    {
        "id": "pt-4",
        "original_point_ids": ["w2-p2"],
        "polished_text": (
            "During the database reliability deep dive, Sarah Jenkins presented metrics illustrating severe connection "
            "pool saturation on the primary Aurora PostgreSQL instance during morning peak spikes. With over eighty backend "
            "microservice pods each maintaining twenty idle connections, the total active connection count exceeded the "
            "database connection limit, triggering connection queuing and transient HTTP 504 gateway timeouts. To eliminate "
            "the connection bottleneck and avoid connection starvation, Sarah Jenkins officially took ownership to implement "
            "and deploy PgBouncer connection pooling middleware in transaction mode by November 15, 2026, ensuring that "
            "maximum backend server connections to PostgreSQL are capped at 150 pooled connections while transparently "
            "supporting microservice scaling."
        ),
        "speaker": "Sarah Jenkins",
        "speakers": ["Sarah Jenkins"],
        "action_owner": "Sarah Jenkins",
        "action_items": [],
        "decisions": ["Deploy PgBouncer middleware in transaction mode"],
        "technical_terms": ["Aurora PostgreSQL", "PgBouncer", "connection pooling", "HTTP 504 gateway timeout"],
        "dates": ["November 15, 2026"],
        "numbers": ["80 microservice pods", "20 connections", "150 pooled connections"],
        "timeline_start": 545.0,
        "timeline_end": 720.0,
    },
    {
        "id": "pt-5",
        "original_point_ids": ["w3-p1"],
        "polished_text": (
            "Liam Patel facilitated a comprehensive incident retrospective examining the forty-minute partial outage "
            "that occurred earlier this month. The root-cause analysis revealed that an unannounced upstream network "
            "partition caused packet loss between Kubernetes worker nodes and AWS Route 53 resolvers, which overwhelmed "
            "the localized CoreDNS daemonsets with exponential DNS request retries. The engineering discussion thoroughly "
            "reviewed the sequence of events, observing that the automatic circuit breakers in the HTTP client library "
            "functioned correctly to prevent cascading failures across downstream payment services. Although participants "
            "discussed theoretical benefits of deploying NodeLocal DNSCache to absorb lookup surges, the meeting concluded "
            "that existing mitigation safeguards were acceptable, and no new implementation items were approved."
        ),
        "speaker": "Liam Patel",
        "speakers": ["Liam Patel"],
        "action_owner": None,
        "action_items": [],
        "decisions": ["No changes required for DNS architecture"],
        "technical_terms": ["AWS Route 53", "CoreDNS", "NodeLocal DNSCache", "circuit breaker", "daemonset"],
        "dates": [],
        "numbers": ["40-minute outage"],
        "timeline_start": 725.0,
        "timeline_end": 900.0,
    },
    {
        "id": "pt-6",
        "original_point_ids": ["w3-p2"],
        "polished_text": (
            "Elena Rostova raised critical security and maintainability concerns regarding the legacy session-cookie "
            "authentication flow currently employed across the single-page web portal. Security audits flagged that the "
            "legacy mechanism remains vulnerable to cross-site request forgery when embedded in third-party mobile webviews. "
            "After analyzing the security architecture and compliance requirements, Elena Rostova agreed to take full "
            "ownership to refactor the frontend authentication library to implement the OAuth2 Authorization Code flow with "
            "Proof Key for Code Exchange (PKCE). Elena will overhaul the token storage mechanism to utilize secure in-memory "
            "storage and configure automated refresh token rotation across all client interfaces, though no strict target "
            "delivery date was established during the session."
        ),
        "speaker": "Elena Rostova",
        "speakers": ["Elena Rostova"],
        "action_owner": "Elena Rostova",
        "action_items": [],
        "decisions": ["Adopt OAuth2 PKCE for single-page web portal"],
        "technical_terms": ["OAuth2", "PKCE", "CSRF", "refresh token rotation"],
        "dates": [],
        "numbers": [],
        "timeline_start": 905.0,
        "timeline_end": 1080.0,
    },
    {
        "id": "pt-7",
        "original_point_ids": ["w4-p1"],
        "polished_text": (
            "Carlos Gomez summarized the ongoing telemetry evaluation examining OpenTelemetry collector agents versus "
            "vendor-proprietary Datadog daemons for distributed tracing in staging environments. The team reviewed overhead "
            "metrics from synthetic load testing, which demonstrated that OpenTelemetry introduced approximately 3% CPU "
            "overhead compared to 2.5% for the Datadog native agent, while OpenTelemetry provided vendor neutrality and "
            "native support for the W3C Trace Context standard. Several engineers expressed preference for maintaining vendor "
            "flexibility despite the minor configuration complexity of OpenTelemetry sampling processors. The discussion "
            "served strictly as an informational exchange to align team perspectives on industry trends, and the group "
            "unanimously agreed to postpone any vendor switching decisions until next fiscal year."
        ),
        "speaker": "Carlos Gomez",
        "speakers": ["Carlos Gomez"],
        "action_owner": None,
        "action_items": [],
        "decisions": ["Postpone telemetry vendor decision until next fiscal year"],
        "technical_terms": ["OpenTelemetry", "Datadog", "W3C Trace Context", "sampling processors"],
        "dates": [],
        "numbers": ["3% CPU overhead", "2.5% CPU overhead"],
        "timeline_start": 1085.0,
        "timeline_end": 1260.0,
    },
    {
        "id": "pt-8",
        "original_point_ids": ["w4-p2"],
        "polished_text": (
            "The platform reliability discussion shifted to recurring false-positive alerts that frequently page on-call "
            "engineers outside normal business hours. During the past month, the Prometheus p99 latency alerts on the public "
            "API gateway triggered forty-two times due to brief network blips that resolved themselves within sixty seconds. "
            "The team reached an agreement that the Prometheus alert threshold must be adjusted from the current 500ms "
            "single-sample rule to a sustained three-minute rolling median of 750ms, and the operational incident runbook "
            "must be thoroughly updated with step-by-step triage procedures. However, nobody volunteered or was assigned "
            "to execute this task, leaving the responsibility open and unassigned for the backlog triage meeting."
        ),
        "speaker": "Dave Miller",
        "speakers": ["Dave Miller"],
        "action_owner": None,
        "action_items": [],
        "decisions": ["Update Prometheus latency alert thresholds and incident runbook"],
        "technical_terms": ["Prometheus", "p99 latency", "API gateway", "rolling median", "runbook triage"],
        "dates": [],
        "numbers": ["42 times", "60 seconds", "500ms", "750ms", "3-minute rolling median"],
        "timeline_start": 1265.0,
        "timeline_end": 1440.0,
    },
    {
        "id": "pt-9",
        "original_point_ids": ["w5-p1"],
        "polished_text": (
            "Rachel Green shared the quarterly infrastructure financial assessment analyzing cloud infrastructure spend "
            "across all AWS accounts. Compute expenditure fell by 14% month-over-month following the successful decommissioning "
            "of legacy EC2 m4.large instances and migrating batch compute workloads to Graviton3 instances. The automated "
            "reservation management platform achieved an overall Reserved Instance and Savings Plan coverage rate of 88%, "
            "which aligns well within corporate efficiency guidelines. The finance department expressed satisfaction with "
            "the current margin trajectory and confirmed that the existing infrastructure budget allocation remains unchanged "
            "for the subsequent financial quarter, concluding the informational review with no follow-up requests or adjustments "
            "requested from the engineering organization."
        ),
        "speaker": "Rachel Green",
        "speakers": ["Rachel Green"],
        "action_owner": None,
        "action_items": [],
        "decisions": ["Maintain current infrastructure budget"],
        "technical_terms": ["AWS EC2", "Graviton3", "Reserved Instances", "Savings Plans"],
        "dates": [],
        "numbers": ["14% reduction", "88% coverage"],
        "timeline_start": 1445.0,
        "timeline_end": 1620.0,
    },
    {
        "id": "pt-10",
        "original_point_ids": ["w5-p2"],
        "polished_text": (
            "Vikram Patel presented an informational briefing regarding industry security trends and the eventual sunsetting "
            "of legacy TLS 1.1 and weak CBC-mode cipher suites across external reverse proxies. Client telemetry over the "
            "previous ninety days revealed that legacy cipher traffic accounted for fewer than 0.02% of all inbound secure "
            "handshakes, with almost all traffic originating from deprecated mobile clients that have already received "
            "end-of-life notices. The security working group discussed the potential compatibility impact on third-party API "
            "consumers and agreed that modernizing cryptographic configurations aligns with enterprise security policies. The "
            "topic was logged strictly as contextual background information for upcoming annual compliance audits, without "
            "establishing any work tickets or scheduling enforcement phases."
        ),
        "speaker": "Vikram Patel",
        "speakers": ["Vikram Patel"],
        "action_owner": None,
        "action_items": [],
        "decisions": ["Acknowledge TLS 1.1 deprecation roadmap"],
        "technical_terms": ["TLS 1.1", "CBC-mode ciphers", "reverse proxy", "cryptographic handshake"],
        "dates": [],
        "numbers": ["90 days", "0.02% traffic"],
        "timeline_start": 1625.0,
        "timeline_end": 1800.0,
    },
]


def test_input_points_meet_length_requirement():
    """Verify all 10 points contain more than 100 words."""
    assert len(STAGE_2_POINTS) == 10
    for idx, p in enumerate(STAGE_2_POINTS, 1):
        word_count = len(p["polished_text"].split())
        assert word_count > 100, f"Point {idx} ({p['id']}) has {word_count} words; must be > 100."


def test_action_point_generation_current_method():
    """Run the 10 Stage 2 points through current extract_actions_from_enhanced_points method."""
    provider = get_provider()
    actions = provider.extract_actions_from_enhanced_points(STAGE_2_POINTS)

    print(f"\nExtracted {len(actions)} action items:")
    for a in actions:
        print(f"- [{a.get('source_point_id')}] Owner: {a.get('owner')} | Deadline: {a.get('deadline')} | Task: {a.get('task')}")

    # Verify action points were extracted
    assert isinstance(actions, list)
    assert len(actions) > 0

    # Verify generated Task contains the corresponding action, owner, and deadline
    actions_by_id = {}
    for a in actions:
        actions_by_id.setdefault(a.get("source_point_id"), []).append(a)

    # Point 2: Marcus Vance committed to Redis 7.2 migration by October 24, 2026
    assert "pt-2" in actions_by_id, "pt-2 should generate an action item"
    pt2_task = actions_by_id["pt-2"][0]["task"]
    assert any(name in pt2_task for name in ["Marcus", "Marcus Vance"]), f"pt-2 task should contain owner: {pt2_task}"
    assert "Redis" in pt2_task, f"pt-2 task should contain Redis action: {pt2_task}"
    assert any(d in pt2_task for d in ["October 24", "October 24, 2026", "2026"]), f"pt-2 task should contain deadline: {pt2_task}"

    # Point 4: Sarah Jenkins committed to PgBouncer deployment by November 15, 2026
    assert "pt-4" in actions_by_id, "pt-4 should generate an action item"
    pt4_task = actions_by_id["pt-4"][0]["task"]
    assert any(name in pt4_task for name in ["Sarah", "Sarah Jenkins"]), f"pt-4 task should contain owner: {pt4_task}"
    assert any(kw in pt4_task for kw in ["PgBouncer", "connection pool", "pooling"]), f"pt-4 task should contain action: {pt4_task}"
    assert any(d in pt4_task for d in ["November 15", "November 15, 2026", "2026"]), f"pt-4 task should contain deadline: {pt4_task}"

    # Point 6: Elena Rostova agreed to refactor auth library (OAuth2 PKCE, no deadline)
    assert "pt-6" in actions_by_id, "pt-6 should generate an action item"
    pt6_task = actions_by_id["pt-6"][0]["task"]
    assert any(name in pt6_task for name in ["Elena", "Elena Rostova"]), f"pt-6 task should contain owner: {pt6_task}"
    assert any(kw in pt6_task for kw in ["OAuth2", "PKCE", "authentication", "auth"]), f"pt-6 task should contain action: {pt6_task}"


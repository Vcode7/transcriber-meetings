import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_test")

from services.ai_provider import get_provider
from services.rom_service import RomService

# 12 Long Discussion Points (> 100 words each)
TWELVE_LONG_POINTS = [
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
    {
        "id": "pt-11",
        "polished_text": (
            "Tariq Mansour presented an urgent stability review of our centralized Elasticsearch log analysis cluster, which recently triggered multiple P1 "
            "cluster health yellow alerts during weekend indexing spikes. Diagnostics revealed that JVM garbage collection pauses exceeded twenty-two seconds "
            "due to excessive field data cache accumulation and over five hundred unassigned shard replicas distributed across fifteen data nodes. To restore "
            "cluster stability and establish predictable memory overhead, Tariq Mansour agreed to overhaul index lifecycle management (ILM) policies by October 30, "
            "2026. Tariq confirmed he will enforce automatic rollover at thirty gigabytes per shard, migrate cold indices to compressed object storage after fourteen "
            "days, and set heap memory limits to prevent runaway JVM allocation."
        ),
        "speaker": "Tariq Mansour",
        "speakers": ["Tariq Mansour"],
        "action_owner": "Tariq Mansour",
    },
    {
        "id": "pt-12",
        "polished_text": (
            "Chloe Dupont led a retrospective on continuous integration bottlenecks affecting the developer velocity across forty-five backend engineering teams. "
            "Average pipeline execution duration climbed from twelve minutes to over thirty-eight minutes over the past two quarters because ephemeral GitLab "
            "runners were downloading large multi-stage Docker base images from scratch on every pull request commit. Following benchmark tests with remote layer "
            "caching, Chloe Dupont committed to deploying a shared, dedicated BuildKit caching daemon cluster with high-performance NVMe storage volumes by "
            "November 20, 2026. Chloe confirmed that the new infrastructure will cut Docker build times by sixty-five percent and provide distributed cache hit "
            "rates above eighty percent across all microservice repositories."
        ),
        "speaker": "Chloe Dupont",
        "speakers": ["Chloe Dupont"],
        "action_owner": "Chloe Dupont",
    },
]

AGENDA_TITLE = "Platform Infrastructure, Cloud Operations & Reliability Engineering"

def main():
    provider = get_provider()
    rom_service = RomService()

    final_rom_input = {
        "meeting_date": "2026-09-21",
        "meeting_time": "10:00 AM",
        "agendas": [
            {
                "agenda_id": "A1",
                "title": AGENDA_TITLE,
                "discussion_points": TWELVE_LONG_POINTS,
            }
        ],
    }

    # Intercept provider._infer to capture prompts, outputs, and reasoning
    captured_calls = []
    original_infer = provider._infer

    def hooked_infer(prompt, max_new_tokens=4096, task_key=None, think=None):
        logger.info(f"--> [HOOKED INFER] task_key={task_key}, max_new_tokens={max_new_tokens}, think={think}")
        output = original_infer(prompt, max_new_tokens=max_new_tokens, task_key=task_key, think=think)
        captured_calls.append({
            "task_key": task_key,
            "max_new_tokens": max_new_tokens,
            "think": think,
            "prompt_length": len(prompt),
            "output": output,
        })
        logger.info(f"<-- [HOOKED INFER] received output length={len(output) if output else 0}")
        return output

    provider._infer = hooked_infer

    print("================================================================================")
    print("STEP 1: RUNNING SHORT ROM GENERATION - REASONING MODE OFF (NORMAL PIPELINE)")
    print("================================================================================")
    captured_calls.clear()
    rom_off = rom_service.generate_rom_version(final_rom_input, version="short", reasoning_mode=False)
    off_calls = list(captured_calls)

    print("================================================================================")
    print("STEP 2: RUNNING SHORT ROM GENERATION - REASONING MODE ON (2-CALL PIPELINE)")
    print("================================================================================")
    captured_calls.clear()
    rom_on = rom_service.generate_rom_version(final_rom_input, version="short", reasoning_mode=True)
    on_calls = list(captured_calls)

    # Save detailed artifacts for analysis
    results_payload = {
        "agenda_title": AGENDA_TITLE,
        "input_points": TWELVE_LONG_POINTS,
        "off_calls": off_calls,
        "on_calls": on_calls,
        "output_off": rom_off["agendas"][0]["discussion_points"],
        "output_on": rom_on["agendas"][0]["discussion_points"],
    }

    out_file = os.path.join(os.path.dirname(__file__), "live_short_rom_test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"Results successfully written to {out_file}")

if __name__ == "__main__":
    main()

"""
Admin License Generator.
Signs license payloads using the Ed25519 private key.
THIS SCRIPT IS FOR ADMIN/DEVELOPMENT USE ONLY AND MUST NEVER BE DELIVERED TO CLIENTS.
"""
import argparse
import base64
from datetime import date, datetime
import json
from pathlib import Path
import sys

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

ADMIN_DIR = Path(__file__).resolve().parent
PRIVATE_KEY_PATH = ADMIN_DIR / "private_key.pem"
PUBLIC_KEY_PATH = ADMIN_DIR / "public_key.pem"

try:
    from generate_keys import generate_keypair
except ImportError:
    from admin_license.generate_keys import generate_keypair


def get_or_create_keys() -> tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
    if not PRIVATE_KEY_PATH.exists() or not PUBLIC_KEY_PATH.exists():
        generate_keypair()

    with open(PRIVATE_KEY_PATH, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    with open(PUBLIC_KEY_PATH, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

    return private_key, public_key


def canonical_bytes(payload: dict) -> bytes:
    """Return deterministic canonical UTF-8 bytes for JSON signing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def create_signed_license(
    customer: str,
    license_type: str,
    expiry_date: str,
    machine_id: str | None = None,
    features: list[str] | None = None,
    issued_date: str | None = None,
    output_path: str | Path | None = None,
) -> dict:
    private_key, _ = get_or_create_keys()

    # Validate expiry date format
    datetime.strptime(expiry_date, "%Y-%m-%d")

    if not issued_date:
        issued_date = date.today().isoformat()

    if features is None:
        features = ["all"]

    clean_machine_id = machine_id.strip().upper() if machine_id and machine_id.strip().upper() != "ANY" else None

    payload = {
        "version": 1,
        "customer": customer.strip(),
        "license_type": license_type.strip().lower(),
        "issued_date": issued_date,
        "expiry_date": expiry_date.strip(),
        "machine_id": clean_machine_id,
        "features": features,
    }

    # Sign canonical payload
    data_to_sign = canonical_bytes(payload)
    signature_bytes = private_key.sign(data_to_sign)
    signature_b64 = base64.b64encode(signature_bytes).decode("ascii")

    license_document = {
        "payload": payload,
        "signature": signature_b64,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(license_document, f, indent=2)
        print(f"[Admin] License successfully written to: {out.resolve()}")

    return license_document


def verify_license_file(license_path: str | Path, public_key_path: str | Path | None = None) -> tuple[bool, str, dict | None]:
    """Verify a license file against the public key."""
    pub_path = Path(public_key_path) if public_key_path else PUBLIC_KEY_PATH
    with open(pub_path, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

    with open(license_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    payload = doc.get("payload")
    sig_b64 = doc.get("signature")
    if not payload or not sig_b64:
        return False, "Malformed license file: missing payload or signature", None

    try:
        sig_bytes = base64.b64decode(sig_b64)
        data = canonical_bytes(payload)
        public_key.verify(sig_bytes, data)
    except Exception as e:
        return False, f"Signature verification failed: {e}", payload

    # Check expiry
    try:
        exp = datetime.strptime(payload["expiry_date"], "%Y-%m-%d").date()
        today = date.today()
        if today > exp:
            return False, f"License expired on {exp.isoformat()} (today is {today.isoformat()})", payload
    except Exception as e:
        return False, f"Invalid expiry date in payload: {e}", payload

    return True, "License is valid and verified", payload


def main():
    parser = argparse.ArgumentParser(description="Admin License Generator for VoiceSum")
    parser.add_argument("--customer", default="Offline Client Demo", help="Customer or organization name")
    parser.add_argument("--type", default="demo", choices=["demo", "trial", "commercial"], help="License type")
    parser.add_argument("--expiry", default="2026-10-15", help="Expiry date (YYYY-MM-DD)")
    parser.add_argument("--machine-id", default=None, help="Optional client Machine ID to bind license to a single host")
    parser.add_argument("--features", default="all", help="Comma-separated features, e.g. 'all' or 'transcription,mom,rag'")
    parser.add_argument("--output", default=None, help="Output path for license.key")
    parser.add_argument("--verify", default=None, help="Verify an existing license file")

    args = parser.parse_args()

    if args.verify:
        valid, msg, payload = verify_license_file(args.verify)
        status = "VALID" if valid else "INVALID"
        print(f"Verification: [{status}] {msg}")
        if payload:
            print(json.dumps(payload, indent=2))
        sys.exit(0 if valid else 1)

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    output_path = args.output or (ADMIN_DIR.parent / "backend" / "license.key")

    print("[Admin] Creating signed license:")
    print(f"  Customer:     {args.customer}")
    print(f"  Type:         {args.type}")
    print(f"  Expiry:       {args.expiry}")
    print(f"  Machine ID:   {args.machine_id or 'Floating (unbound)'}")
    print(f"  Features:     {features}")
    print(f"  Output:       {output_path}")

    create_signed_license(
        customer=args.customer,
        license_type=args.type,
        expiry_date=args.expiry,
        machine_id=args.machine_id,
        features=features,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

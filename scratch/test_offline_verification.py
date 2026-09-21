"""
Comprehensive License & Offline Verification Suite.
Executes the four mandatory license tests:
  Test 1: Valid license (Expiry: 2026-10-15) -> Application works.
  Test 2: Tampered license (Modified expiry date) -> Signature verification fails -> Application blocked.
  Test 3: Expired license -> License expired -> Application blocked.
  Test 4: Machine mismatch -> Machine/license mismatch -> Application blocked.
"""
from datetime import date
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
ADMIN_DIR = ROOT_DIR / "admin_license"
DEPLOYMENT_DIR = ROOT_DIR / "deployment"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR))

import license
from admin_license.generate_license import create_signed_license


def run_tests():
    print("=" * 70)
    print("STARTING MANDATORY LICENSE TESTS")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: Valid License (Expiry: 2026-10-15)
    # -------------------------------------------------------------
    print("\n--- TEST 1: Valid License (Expiry: 2026-10-15) ---")
    valid_lic_path = DEPLOYMENT_DIR / "license.key"
    assert valid_lic_path.exists(), "deployment/license.key missing!"

    # Ensure backend uses this key
    shutil.copy2(valid_lic_path, BACKEND_DIR / "license.key")
    valid, msg = license.check_license()
    info = license.get_license_info()
    print(f"Result: Valid={valid}, Message='{msg}'")
    print(f"License Info: {info}")
    assert valid is True, f"Test 1 FAILED: Expected valid=True, got {valid} with msg: {msg}"
    assert info.get("expiry_date") == "2026-10-15", f"Expected expiry 2026-10-15, got {info.get('expiry_date')}"
    print("[PASS] Test 1: Valid license successfully authorized.")

    # -------------------------------------------------------------
    # Test 2: Tampered Expiry Date
    # -------------------------------------------------------------
    print("\n--- TEST 2: Tampered License (Client attempts to extend demo) ---")
    # Read valid license
    with open(valid_lic_path, "r", encoding="utf-8") as f:
        tampered_doc = json.load(f)

    # Client maliciously changes expiry from 2026-10-15 to 2028-12-31 without re-signing
    tampered_doc["payload"]["expiry_date"] = "2028-12-31"

    tampered_path = BACKEND_DIR / "license.key"
    with open(tampered_path, "w", encoding="utf-8") as f:
        json.dump(tampered_doc, f, indent=2)

    valid, msg = license.check_license()
    print(f"Result: Valid={valid}, Message='{msg}'")
    assert valid is False, "Test 2 FAILED: Tampered license was accepted!"
    assert "signature verification failed" in msg.lower(), f"Expected signature failure message, got: {msg}"
    print("[PASS] Test 2: Tampered license detected. Application blocked.")

    # -------------------------------------------------------------
    # Test 3: Expired License
    # -------------------------------------------------------------
    print("\n--- TEST 3: Expired License (e.g. expired on 2026-09-01) ---")
    expired_lic_path = BACKEND_DIR / "license.key"
    create_signed_license(
        customer="Expired Client",
        license_type="demo",
        expiry_date="2026-09-01",
        output_path=expired_lic_path,
    )

    valid, msg = license.check_license()
    print(f"Result: Valid={valid}")
    print(f"Message:\n{msg}")
    assert valid is False, "Test 3 FAILED: Expired license was accepted!"
    assert "expired" in msg.lower(), f"Expected expired message, got: {msg}"
    assert "2026-09-01" in msg, f"Expected expiry date in message, got: {msg}"
    print("[PASS] Test 3: Expired license properly rejected.")

    # -------------------------------------------------------------
    # Test 4: Machine ID Binding Mismatch
    # -------------------------------------------------------------
    print("\n--- TEST 4: Machine Mismatch (Node-locked to different hardware) ---")
    fake_machine_id = "11112222-3333-4444-5555-666677778888"
    mismatch_lic_path = BACKEND_DIR / "license.key"
    create_signed_license(
        customer="Bound Client",
        license_type="demo",
        expiry_date="2026-10-15",
        machine_id=fake_machine_id,
        output_path=mismatch_lic_path,
    )

    valid, msg = license.check_license()
    print(f"Result: Valid={valid}")
    print(f"Message:\n{msg}")
    assert valid is False, "Test 4 FAILED: License bound to another machine was accepted!"
    assert "mismatch" in msg.lower(), f"Expected mismatch message, got: {msg}"
    print("[PASS] Test 4: Machine ID mismatch detected. Application blocked.")

    # -------------------------------------------------------------
    # Restore Valid Demo License
    # -------------------------------------------------------------
    print("\n--- RESTORING VALID DEMO LICENSE ---")
    shutil.copy2(valid_lic_path, BACKEND_DIR / "license.key")
    valid, msg = license.check_license()
    assert valid is True
    print(f"[RESTORED] Backend license.key restored to valid demo license ({license.get_license_info()['expiry_date']}).")

    print("\n" + "=" * 70)
    print("ALL 4 MANDATORY LICENSE TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

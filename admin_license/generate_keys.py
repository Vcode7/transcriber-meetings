"""
Admin Keypair Generator.
Generates an Ed25519 private key (admin only) and public key (for verification).
This script and private_key.pem MUST REMAIN STRICTLY ON THE ADMIN/DEVELOPER MACHINE.
"""
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

ADMIN_DIR = Path(__file__).resolve().parent
PRIVATE_KEY_PATH = ADMIN_DIR / "private_key.pem"
PUBLIC_KEY_PATH = ADMIN_DIR / "public_key.pem"


def generate_keypair(force: bool = False) -> tuple[Path, Path]:
    if PRIVATE_KEY_PATH.exists() and not force:
        print(f"[Admin] Existing private key found at: {PRIVATE_KEY_PATH}")
        return PRIVATE_KEY_PATH, PUBLIC_KEY_PATH

    print("[Admin] Generating new Ed25519 keypair...")
    private_key = ed25519.Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    with open(PRIVATE_KEY_PATH, "wb") as f:
        f.write(private_bytes)

    with open(PUBLIC_KEY_PATH, "wb") as f:
        f.write(public_bytes)

    print(f"[Admin] Private key saved to: {PRIVATE_KEY_PATH} (DO NOT DISTRIBUTE!)")
    print(f"[Admin] Public key saved to:  {PUBLIC_KEY_PATH}")
    return PRIVATE_KEY_PATH, PUBLIC_KEY_PATH


if __name__ == "__main__":
    generate_keypair()

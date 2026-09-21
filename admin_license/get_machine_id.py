"""
Machine ID Extractor.
Generates a deterministic, hardware-bound unique identifier for the host machine.
Can be run on client machines to provide their Machine ID to the administrator for node-locked licenses.
"""
import hashlib
import platform
import subprocess
import uuid
import sys


def get_machine_id() -> str:
    """
    Generate a stable SHA-256 machine identifier based on local hardware/OS traits.
    Does not require internet access.
    """
    parts = [
        platform.system(),
        platform.machine(),
        str(uuid.getnode()),  # MAC address integer
    ]

    if platform.system() == "Windows":
        # 1. Windows MachineGuid from Registry
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                guid, _ = winreg.QueryValueEx(key, "MachineGuid")
                if guid:
                    parts.append(str(guid).strip())
        except Exception:
            pass

        # 2. Motherboard / BIOS UUID via PowerShell
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-CimInstance -Class Win32_ComputerSystemProduct).UUID"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5
            ).strip()
            if out and "error" not in out.lower():
                parts.append(out)
        except Exception:
            pass
    elif platform.system() == "Linux":
        for p in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    parts.append(f.read().strip())
                    break
            except Exception:
                pass

    raw = "::".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    # Format as 8-4-4-4-12 hex string
    formatted = f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}".upper()
    return formatted


if __name__ == "__main__":
    machine_id = get_machine_id()
    print("=" * 60)
    print("HOST MACHINE ID:")
    print(f"  {machine_id}")
    print("=" * 60)
    print("Provide this ID to the administrator to request a node-locked license.")

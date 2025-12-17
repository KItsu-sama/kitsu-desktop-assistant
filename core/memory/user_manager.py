"""
user_management.py

Full User Management System for Kitsu (local, file-based).
Features:
- UserManager (load/save JSON per-profile)
- Thread-safe (RLock)
- Field normalization and allowed-fields enforcement
- set_user_info / get_user_info / reset_user_info
- Creator fingerprint generation and verification (locks Creator status)
- CLI-friendly helpers for your existing `/user` command
- Safe defaults and audit logging

Drop this into /core/user_manager.py and import it from your main orchestrator.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import platform
import uuid
import subprocess
import datetime


DEFAULT_USER = {
    "name": "User",
    "nickname": "",
    "refer_title": "User",
    "status": "User", 
    "gender": "Unknown",
    "relationship": {
        "role": "User",
        "trust_level": 0.1,
        "affinity": 1.0,
        "lore_tag": ""
    },
    "permissions": {
        "is_admin": False,
        "can_shutdown": False,
        "can_edit_memory": False,
        "can_trigger_modes": False,
        "can_toggle_avatar": False
    },
    "preferences": {
        "preferred_kitsu_style": "neutral",
        "preferred_kitsu_mood": "neutral",
        "avatar_enabled": True,
        "voice_output": True,
        "emotion_reactions": True
    },
    "stats": {
        "first_seen": None,
        "last_seen": None,
        "messages_sent": 0
    },
    # creator_fingerprint may be set on first run for the real creator machine
    "creator_fingerprint": None
}


# Map user-provided field names to canonical keys stored in JSON
FIELD_MAP = {
    "name": "name",
    "nickname": "nickname",
    "nick": "nickname",
    "title": "refer_title",
    "refer_title": "refer_title",
    "gender": "gender",
    "status": "status",
    "role": "status",
    "relationship": "relationship",
    "permissions": "permissions",
    "preference": "preferences",
    "preferences": "preferences",
    "preference_kitus_style": "preferences",
}

# Allowed top-level writable fields via the /user set command
ALLOWED_FIELDS = {
    "name",
    "nickname",
    "gender",
    "refer_title", 
    "relationship",
    "permissions",
    "preferences",
}

# Fields that are locked and cannot be changed by /user set
LOCKED_FIELDS = {
    "status",
    "creator_fingerprint",
}


class UserManager:
    """
    Manage per-user JSON profiles safely.

    Usage:
        um = UserManager(data_dir=Path("data/config/users"))
        um.load_profile("default")
        info = um.get_user_info()
        um.set_user_info(name="Zino")

    The class is thread-safe using an RLock.
    """

    def __init__(self, data_dir: Path, autosave: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.autosave = autosave

        self.profile_name: Optional[str] = None
        self.profile_path: Optional[Path] = None
        self._state: Dict[str, Any] = {}

    # -----------------------------
    # Low-level load/save helpers
    # -----------------------------
    def _profile_path_for(self, name: str) -> Path:
        safe_name = name.replace("/", "_")
        return self.data_dir / f"{safe_name}.json"

    def load_profile(self, name: str = "default") -> None:
        """Load (or create) a profile by name."""
        with self._lock:
            self.profile_name = name
            self.profile_path = self._profile_path_for(name)

            if not self.profile_path.exists():
                # create default profile
                now = datetime.datetime.utcnow().isoformat()
                p = dict(DEFAULT_USER)
                p["stats"] = dict(DEFAULT_USER["stats"])  # copy
                p["stats"]["first_seen"] = now
                p["stats"]["last_seen"] = now
                self._state = p
                self._save()
                return

            # load existing
            with open(self.profile_path, "r", encoding="utf-8") as f:
                try:
                    self._state = json.load(f)
                except Exception:
                    # corrupted file → overwrite with default (backup first)
                    backup = self.profile_path.with_suffix(".corrupt.json")
                    self.profile_path.rename(backup)
                    now = datetime.datetime.utcnow().isoformat()
                    p = dict(DEFAULT_USER)
                    p["stats"] = dict(DEFAULT_USER["stats"])  # copy
                    p["stats"]["first_seen"] = now
                    p["stats"]["last_seen"] = now
                    self._state = p
                    self._save()

            # run validation
            self._ensure_defaults()
            # Validate creator fingerprint (if status==Creator)
            self._validate_creator_status()

    def _save(self) -> None:
        if self.profile_path is None:
            raise RuntimeError("No profile loaded")
        with self._lock:
            tmp = self.profile_path.with_suffix(".tmp.json")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
            tmp.replace(self.profile_path)

    def _ensure_defaults(self) -> None:
        """Ensure any missing fields are populated with sensible defaults."""
        with self._lock:
            # shallow merge DEFAULT_USER for missing top-level keys
            for k, v in DEFAULT_USER.items():
                if k not in self._state:
                    self._state[k] = v
                else:
                    # ensure nested dicts exist
                    if isinstance(v, dict) and not isinstance(self._state.get(k), dict):
                        self._state[k] = v

    # -----------------------------
    # Fingerprint utilities
    # -----------------------------
    @staticmethod
    def generate_machine_fingerprint() -> str:
        """Create a hashed fingerprint using several local non-sensitive identifiers.

        This avoids storing raw serials; we hash the concatenation.
        Works cross-platform with best-effort data.
        """
        parts = []
        # uuid.getnode() returns mac-based node (may be stable)
        try:
            parts.append(str(uuid.getnode()))
        except Exception:
            pass

        # platform info
        try:
            parts.append(platform.node())
            parts.append(platform.platform())
        except Exception:
            pass

        # attempt to read a stable machine id on linux
        try:
            if Path("/etc/machine-id").exists():
                parts.append(Path("/etc/machine-id").read_text().strip())
        except Exception:
            pass

        # windows serial (best-effort, not required)
        try:
            if platform.system().lower().startswith("win"):
                out = subprocess.check_output("wmic csproduct get uuid", shell=True)
                out = out.decode(errors="ignore").strip().splitlines()
                if len(out) > 1:
                    parts.append(out[1].strip())
        except Exception:
            pass

        raw = "|".join([p for p in parts if p])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _validate_creator_status(self) -> None:
        """If profile claims Creator status, ensure fingerprint matches. If missing, set it on first run."""
        with self._lock:
            status = str(self._state.get("status", "User"))
            stored_fp = self._state.get("creator_fingerprint")
            current_fp = self.generate_machine_fingerprint()

            if status.lower() == "creator":
                if stored_fp is None:
                    # first run claiming creator — bind it to this machine
                    self._state["creator_fingerprint"] = current_fp
                    # ensure admin permissions
                    self._state.setdefault("permissions", {})
                    self._state["permissions"].update({
                        "is_admin": True,
                        "can_shutdown": True,
                        "can_edit_memory": True,
                        "can_trigger_modes": True,
                        "can_toggle_avatar": True,
                    })
                    if self.autosave:
                        self._save()
                    return

                # if fingerprint mismatch, downgrade silently and log
                if stored_fp != current_fp:
                    # downgrade profile
                    self._state["status"] = "User"
                    self._state["refer_title"] = "User"
                    # reset permissions to defaults
                    self._state["permissions"] = dict(DEFAULT_USER["permissions"])
                    # clear creator_fingerprint to avoid future confusion
                    # keep the previous stored_fp for audit (store in logs elsewhere if desired)
                    if self.autosave:
                        self._save()

    # -----------------------------
    # Public API: get/set/reset
    # -----------------------------
    def get_user_info(self) -> Dict[str, Any]:
        """Return a safe, canonical user info dict for UI printing."""
        with self._lock:
            self._ensure_defaults()
            s = self._state
            return {
                "name": s.get("name", DEFAULT_USER["name"]),
                "nickname": s.get("nickname", DEFAULT_USER["nickname"]),
                "refer_title": s.get("refer_title", DEFAULT_USER["refer_title"]),
                "status": s.get("status", DEFAULT_USER["status"]),
                "gender": s.get("gender", DEFAULT_USER["gender"]),
                "relationship": s.get("relationship", dict(DEFAULT_USER["relationship"])),
                "permissions": s.get("permissions", dict(DEFAULT_USER["permissions"])),
                "preferences": s.get("preferences", dict(DEFAULT_USER["preferences"])),
                "stats": s.get("stats", dict(DEFAULT_USER["stats"])),
            }

    def set_user_info(self, **updates) -> None:
        """Update user profile with nested merge logic.

        Field names are normalized via FIELD_MAP. Locked fields will raise ValueError.
        Unknown fields are rejected to avoid JSON pollution.
        """
        with self._lock:
            if not updates:
                return

            for raw_key, raw_val in updates.items():
                # normalize dotted keys like relationship.trust_level
                if "." in raw_key:
                    top, sub = raw_key.split(".", 1)
                    key = FIELD_MAP.get(top.lower(), top.lower())

                    if key in LOCKED_FIELDS:
                        raise ValueError(f"Field '{key}' is locked and cannot be changed manually.")

                    if key not in ALLOWED_FIELDS:
                        raise ValueError(f"Unknown or unsupported user field: {top}")

                    self._state.setdefault(key, {})
                    # try to coerce type if numeric/boolean provided as string is handled upstream
                    self._state[key][sub] = raw_val
                    continue

                # normalize top-level field names
                key = FIELD_MAP.get(raw_key.lower(), raw_key.lower())

                if key in LOCKED_FIELDS:
                    raise ValueError(f"Field '{key}' is locked and cannot be changed manually.")

                # only allow a known whitelist of top-level keys
                if key not in ALLOWED_FIELDS:
                    raise ValueError(f"Unknown or unsupported user field: {raw_key}")

                # if nested dict provided, merge
                if isinstance(raw_val, dict) and isinstance(self._state.get(key), dict):
                    self._state[key].update(raw_val)
                else:
                    self._state[key] = raw_val

            # update last_seen
            now = datetime.datetime.utcnow().isoformat()
            self._state.setdefault("stats", {})
            self._state["stats"]["last_seen"] = now

            if self.autosave:
                self._save()

    def reset_user_info(self, target: Optional[str] = None) -> None:
        """Reset profile parts to defaults. target can be 'profile', 'permissions', or None for everything."""
        with self._lock:
            if target is None or target == "all":
                # reset everything to DEFAULT_USER except preserve creator_fingerprint if present
                preserved_fp = self._state.get("creator_fingerprint")
                p = dict(DEFAULT_USER)
                p["stats"] = dict(DEFAULT_USER["stats"])  # copy
                if preserved_fp:
                    p["creator_fingerprint"] = preserved_fp
                self._state = p

            elif target == "permissions":
                self._state["permissions"] = dict(DEFAULT_USER["permissions"]) 

            elif target == "profile":
                keep_fp = self._state.get("creator_fingerprint")
                now = datetime.datetime.utcnow().isoformat()
                p = dict(DEFAULT_USER)
                p["stats"] = dict(DEFAULT_USER["stats"])  # copy
                p["stats"]["first_seen"] = now
                p["stats"]["last_seen"] = now
                if keep_fp:
                    p["creator_fingerprint"] = keep_fp
                self._state = p

            else:
                raise ValueError("Unknown reset target")

            if self.autosave:
                self._save()

    # -----------------------------
    # Helpers for CLI integration
    # -----------------------------
    def handle_user_command(self, command: str) -> str:
        """A convenience wrapper for your /user CLI command.

        Returns a human-friendly message string (safe to print).
        """
        with self._lock:
            parts = command.strip().split()
            if not parts:
                return ""

            if len(parts) == 1:
                # just '/user' -> show
                info = self.get_user_info()
                lines = ["📊 User Info:"]
                lines.append(f"  Name: {info.get('name')}")
                lines.append(f"  Gender: {info.get('gender')}")
                lines.append(f"  Nickname: {info.get('nickname')}")
                lines.append(f"  Title (Kitsu calls you): {info.get('refer_title')}")
                lines.append(f"  Status: {info.get('status')}")
                rel = info.get('relationship', {})
                lines.append(f"  Relationship: trust={rel.get('trust_level')}, affinity={rel.get('affinity')}, lore='{rel.get('lore_tag','')}'")
                perms = info.get('permissions', {})
                perms_list = ', '.join([f"{k}={v}" for k, v in perms.items()])
                lines.append(f"  Permissions: {perms_list}")
                return "\n".join(lines)

            sub = parts[1].lower()
            if sub == 'set' and len(parts) >= 4:
                field = parts[2]
                # preserve original-case value using original command string
                raw_parts = command.strip().split(' ', 3)
                value_raw = raw_parts[3] if len(raw_parts) >= 4 else ''
                # strip quotes if present
                if (value_raw.startswith('"') and value_raw.endswith('"')) or (value_raw.startswith("'") and value_raw.endswith("'")):
                    value_raw = value_raw[1:-1]

                # try to coerce booleans/numbers
                parsed_value: Any = value_raw
                if value_raw.lower() in ['true', 'false']:
                    parsed_value = value_raw.lower() == 'true'
                else:
                    try:
                        if '.' in value_raw:
                            parsed_value = float(value_raw)
                        else:
                            parsed_value = int(value_raw)
                    except Exception:
                        parsed_value = value_raw

                # normalize key
                key = FIELD_MAP.get(field.lower(), field.lower())

                if key in LOCKED_FIELDS:
                    return "❌ This field is locked and cannot be changed manually."

                if key not in ALLOWED_FIELDS:
                    return f"❌ Unknown or unsupported user field: {field}"

                try:
                    # support dotted updates like relationship.trust_level
                    if '.' in field:
                        self.set_user_info(**{field: parsed_value})
                    else:
                        self.set_user_info(**{key: parsed_value})
                except Exception as e:
                    return f"❌ Failed to update user info: {e}"

                return f"✅ Updated user {key} -> {parsed_value}"

            if sub == 'reset':
                what = 'all' if len(parts) == 1 else parts[2].lower()
                if what not in ('profile', 'permissions', 'all'):
                    return "❌ Invalid reset target. Use: profile, permissions, or all"
                self.reset_user_info(None if what == 'all' else what)
                return f"🔁 reset {what} to defaults"

            return f"❌ Unknown /user subcommand: {sub}"


# -----------------------------
# Example usage
# -----------------------------
if __name__ == '__main__':
    # quick local test
    um = UserManager(Path("./data/config/users"))
    um.load_profile("default")
    print(um.handle_user_command('/user'))
    print(um.handle_user_command('/user set name Zino'))
    print(um.handle_user_command('/user set nickname Zi'))
    print(um.handle_user_command('/user set status Creator'))
    print(um.handle_user_command('/user'))

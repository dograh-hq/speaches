from __future__ import annotations

import json
import sys


def main() -> int:
    print(json.dumps({"event": "ready", "sample_rate": 24000, "voices": ["speaker_a"]}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        request_id = request["id"]
        method = request["method"]
        if method == "ping":
            print(json.dumps({"id": request_id, "ok": True, "result": {"pong": True}}), flush=True)
        elif method == "shutdown":
            print(json.dumps({"id": request_id, "ok": True, "result": {"shutdown": True}}), flush=True)
            return 0
        else:
            print(json.dumps({"id": request_id, "ok": False, "error": f"unsupported method: {method}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

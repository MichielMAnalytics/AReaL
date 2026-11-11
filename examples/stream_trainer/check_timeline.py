# SPDX-License-Identifier: Apache-2.0
"""Check coordinator markers from one complete streamed update in a trainer log."""

import argparse
import json
import re
from pathlib import Path


class TimelineCheck:
    def check(self, path: Path) -> list[dict]:
        updates: dict[str, dict[str, list[float]]] = {}
        for line in path.read_text(errors="replace").splitlines():
            if "[StreamTrainer]" not in line:
                continue
            fields = dict(re.findall(r"([a-z_]+)=([^\s,]+)", line))
            if not {"update_id", "event", "elapsed_s"}.issubset(fields):
                continue
            events = updates.setdefault(fields["update_id"], {})
            events.setdefault(fields["event"], []).append(float(fields["elapsed_s"]))
        if not updates:
            raise AssertionError(
                "No StreamTrainer markers found; pass the trainer log, not server log"
            )
        results = []
        for update_id, events in updates.items():
            for name in (
                "backward_start",
                "backward_done",
                "rollout_complete",
                "optimizer_start",
                "optimizer_done",
            ):
                assert name in events, f"{update_id}: missing {name}"
            assert len(events["rollout_complete"]) == 1, update_id
            assert (
                len(events["optimizer_start"]) == len(events["optimizer_done"]) == 1
            ), update_id
            finished = events["rollout_complete"][0]
            first_backward = min(events["backward_done"])
            assert first_backward < finished, (
                f"{update_id}: no backward completed before rollout"
            )
            assert events["optimizer_start"][0] >= max(
                finished, max(events["backward_done"])
            ), update_id
            assert events["optimizer_done"][0] >= events["optimizer_start"][0], (
                update_id
            )
            results.append(
                {
                    "update_id": update_id,
                    "status": "passed",
                    "first_backward_done_s": first_backward,
                    "rollout_complete_s": finished,
                    "optimizer_done_s": events["optimizer_done"][0],
                }
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(json.dumps(TimelineCheck().check(args.log), indent=2))


if __name__ == "__main__":
    main()

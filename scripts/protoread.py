#!/usr/bin/env python3

import json
import sys
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from google.transit import gtfs_realtime_pb2


def compute_stats(feed: gtfs_realtime_pb2.FeedMessage) -> dict[str, int]:
    stats = {
        "entities": len(feed.entity),
        "trip_updates": 0,
        "stop_time_updates": 0,
        "arrivals": 0,
        "departures": 0,
        "vehicle_positions": 0,
    }

    for entity in feed.entity:
        if entity.HasField("trip_update"):
            stats["trip_updates"] += 1

            for stu in entity.trip_update.stop_time_update:
                stats["stop_time_updates"] += 1

                if stu.HasField("arrival"):
                    stats["arrivals"] += 1

                if stu.HasField("departure"):
                    stats["departures"] += 1

        if entity.HasField("vehicle"):
            stats["vehicle_positions"] += 1

    return stats


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print(
            f"Usage: {Path(sys.argv[0]).name} <file.pb> [output.json]",
            file=sys.stderr,
        )
        sys.exit(1)

    pb_file = Path(sys.argv[1])

    if not pb_file.exists():
        print(f"File not found: {pb_file}", file=sys.stderr)
        sys.exit(1)

    feed = gtfs_realtime_pb2.FeedMessage()

    with open(pb_file, "rb") as f:
        feed.ParseFromString(f.read())

    print(json.dumps(compute_stats(feed), indent=2), file=sys.stderr)

    data = MessageToDict(feed)

    if len(sys.argv) == 3:
        output_file = Path(sys.argv[2])
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {output_file}", file=sys.stderr)
    else:
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
        print()


if __name__ == "__main__":
    main()

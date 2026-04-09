import argparse
import time
from typing import List

import lcm
from lcmtypes.lcmt_object_state.lcmt_object_state import lcmt_object_state
from loguru import logger


def _parse_csv_floats(raw: str) -> List[float]:
    if not raw.strip():
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _default_names(prefix: str, n: int) -> List[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _build_message(
    object_name: str,
    positions: List[float],
    velocities: List[float],
    position_names: List[str],
    velocity_names: List[str],
) -> lcmt_object_state:
    msg = lcmt_object_state()
    msg.utime = int(time.time() * 1_000_000)
    msg.object_name = object_name
    msg.num_positions = len(positions)
    msg.num_velocities = len(velocities)
    msg.position = positions
    msg.velocity = velocities
    msg.position_names = position_names
    msg.velocity_names = velocity_names
    return msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish lcmt_object_state over LCM.")
    parser.add_argument("--object-name", default="R_shape_texture")
    parser.add_argument(
        "--channel",
        default=None,
        help="Default: OBJECT_<object-name>_STATE_SIMULATION",
    )
    parser.add_argument("--positions", default="0.0,0.0")
    parser.add_argument("--velocities", default="0.0,0.0")
    parser.add_argument("--position-names", default="")
    parser.add_argument("--velocity-names", default="")
    parser.add_argument(
        "--count", type=int, default=1, help="Number of messages to send"
    )
    parser.add_argument(
        "--period-sec", type=float, default=0.1, help="Delay between messages"
    )
    args = parser.parse_args()

    positions = _parse_csv_floats(args.positions)
    velocities = _parse_csv_floats(args.velocities)
    pos_names = (
        [x.strip() for x in args.position_names.split(",") if x.strip()]
        if args.position_names.strip()
        else _default_names("pos_", len(positions))
    )
    vel_names = (
        [x.strip() for x in args.velocity_names.split(",") if x.strip()]
        if args.velocity_names.strip()
        else _default_names("vel_", len(velocities))
    )

    if len(pos_names) != len(positions):
        raise ValueError("position_names count must match positions count")
    if len(vel_names) != len(velocities):
        raise ValueError("velocity_names count must match velocities count")

    channel = args.channel or f"OBJECT_{args.object_name}_STATE_SIMULATION"
    lc = lcm.LCM()

    for i in range(args.count):
        msg = _build_message(
            object_name=args.object_name,
            positions=positions,
            velocities=velocities,
            position_names=pos_names,
            velocity_names=vel_names,
        )
        lc.publish(channel, msg.encode())
        logger.info(
            "Published {}/{} to {} for object {}",
            i + 1,
            args.count,
            channel,
            args.object_name,
        )
        if i + 1 < args.count:
            time.sleep(max(0.0, args.period_sec))


if __name__ == "__main__":
    main()

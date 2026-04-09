# pyright: reportMissingImports=false

import argparse
import os
import sys
import time
from typing import Optional

import lcm


# These generated LCM Python files use plain imports (e.g., import info2_t),
# so we add their directory directly to sys.path.
_PROCMAN_LCMTYPES_DIR = os.path.join(os.path.dirname(__file__), "lcmtypes", "procman")
if _PROCMAN_LCMTYPES_DIR not in sys.path:
    sys.path.insert(0, _PROCMAN_LCMTYPES_DIR)

import command2_t  # noqa: E402
import info2_t  # noqa: E402
import orders2_t  # noqa: E402
import sheriff_cmd2_t  # noqa: E402


def _capture_latest_host_info(
    lc: lcm.LCM,
    info_channel: str,
    host: str,
    timeout_sec: float,
) -> info2_t.info2_t:
    """Return the newest PMD_INFO2 snapshot for ``host`` seen before the deadline."""
    state = {"last": None}

    def _on_info(_channel: str, data: bytes) -> None:
        msg = info2_t.info2_t.decode(data)
        if msg.host == host:
            state["last"] = msg

    sub = lc.subscribe(info_channel, _on_info)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        lc.handle_timeout(100)
    lc.unsubscribe(sub)

    if state["last"] is None:
        raise RuntimeError(
            "Did not receive {} for host='{}' within {:.1f}s".format(
                info_channel, host, timeout_sec
            )
        )
    return state["last"]


def _find_actual_runid(info: info2_t.info2_t, group: str, command_name: str) -> int:
    for deputy_cmd in info.cmds:
        if (
            deputy_cmd.cmd.group == group
            and deputy_cmd.cmd.command_name == command_name
        ):
            return int(deputy_cmd.actual_runid)
    raise RuntimeError(
        "No deputy command '{}:{}' on host='{}' in latest {} (ncmds={})".format(
            group, command_name, info.host, "PMD_INFO2", info.ncmds
        )
    )


def _copy_command2(c: command2_t.command2_t) -> command2_t.command2_t:
    o = command2_t.command2_t()
    o.exec_str = c.exec_str
    o.command_name = c.command_name
    o.group = c.group
    o.auto_respawn = c.auto_respawn
    o.stop_signal = c.stop_signal
    o.stop_time_allowed = c.stop_time_allowed
    o.num_options = c.num_options
    o.option_names = list(c.option_names)
    o.option_values = list(c.option_values)
    return o


def _build_merged_orders_msg(
    info: info2_t.info2_t,
    host: str,
    sheriff_name: str,
    sheriff_id: int,
    group: str,
    command_name: str,
    exec_str: str,
    action: str,
    desired_runid_override: Optional[int],
) -> orders2_t.orders2_t:
    """
    Rebuild orders from the deputy info snapshot so other commands stay ordered.

    Publishing a partial ``orders2_t`` replaces the full desired set for that host,
    which stops every command not listed.
    """
    sheriff_cmds = []
    for dc in info.cmds:
        cmd_c = _copy_command2(dc.cmd)
        is_target = cmd_c.group == group and cmd_c.command_name == command_name
        sc = sheriff_cmd2_t.sheriff_cmd2_t()
        sc.cmd = cmd_c
        sid = dc.sheriff_id if dc.sheriff_id != 0 else sheriff_id
        if is_target:
            sc.sheriff_id = sheriff_id
            if action == "start":
                sc.cmd.exec_str = exec_str
                sc.desired_runid = (
                    int(desired_runid_override)
                    if desired_runid_override is not None
                    else int(dc.actual_runid) + 1
                )
                sc.force_quit = 0
            else:
                sc.desired_runid = (
                    int(desired_runid_override)
                    if desired_runid_override is not None
                    else int(dc.actual_runid)
                )
                sc.force_quit = 1
        else:
            sc.sheriff_id = sid
            sc.desired_runid = int(dc.actual_runid)
            sc.force_quit = 0
        sheriff_cmds.append(sc)

    msg = orders2_t.orders2_t()
    msg.utime = int(time.time() * 1_000_000)
    msg.host = host
    msg.sheriff_name = sheriff_name
    msg.ncmds = len(sheriff_cmds)
    msg.cmds = sheriff_cmds
    msg.num_options = 0
    msg.option_names = []
    msg.option_values = []
    return msg


def _build_orders_msg(
    host: str,
    sheriff_name: str,
    sheriff_id: int,
    group: str,
    command_name: str,
    exec_str: str,
    desired_runid: int,
    force_quit: int,
) -> "orders2_t.orders2_t":
    cmd = command2_t.command2_t()
    cmd.exec_str = exec_str
    cmd.command_name = command_name
    cmd.group = group
    cmd.auto_respawn = False
    cmd.stop_signal = 2
    cmd.stop_time_allowed = 2.0
    cmd.num_options = 0
    cmd.option_names = []
    cmd.option_values = []

    sheriff_cmd = sheriff_cmd2_t.sheriff_cmd2_t()
    sheriff_cmd.cmd = cmd
    sheriff_cmd.desired_runid = desired_runid
    sheriff_cmd.force_quit = force_quit
    sheriff_cmd.sheriff_id = sheriff_id

    msg = orders2_t.orders2_t()
    msg.utime = int(time.time() * 1_000_000)
    msg.host = host
    msg.sheriff_name = sheriff_name
    msg.ncmds = 1
    msg.cmds = [sheriff_cmd]
    msg.num_options = 0
    msg.option_names = []
    msg.option_values = []
    return msg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start or stop a procman command by publishing orders2_t."
    )
    parser.add_argument(
        "--action",
        choices=("start", "stop"),
        default="start",
        help="start: bump runid; stop: set force_quit (matches bot_procman Sheriff._stop).",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--group", default="debug")
    parser.add_argument("--command-name", default="lcm-spy")
    parser.add_argument(
        "--exec-str", default="bazel-bin/lcmtypes/dair-lcm-spy", help="PMD exec value"
    )
    parser.add_argument("--sheriff-name", default="bot-procman-sheriff")
    parser.add_argument(
        "--sheriff-id",
        type=int,
        default=1,
        help="Must be non-zero; 0 causes bot-procman-sheriff assertion failure.",
    )
    parser.add_argument("--orders-channel", default="PMD_ORDERS2")
    parser.add_argument("--info-channel", default="PMD_INFO2")
    parser.add_argument("--timeout-sec", type=float, default=2.0)
    parser.add_argument(
        "--desired-runid",
        type=int,
        default=None,
        help="Optional override for start (current+1) or stop (usually current actual_runid).",
    )
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help=(
            "Publish only this command (legacy). Without this, merge with latest PMD_INFO2 "
            "so other ordered processes are not stopped."
        ),
    )
    args = parser.parse_args()
    if args.sheriff_id == 0:
        raise ValueError("--sheriff-id must be non-zero")

    lc = lcm.LCM()
    info = _capture_latest_host_info(
        lc=lc,
        info_channel=args.info_channel,
        host=args.host,
        timeout_sec=args.timeout_sec,
    )
    current_runid = _find_actual_runid(
        info, args.group, args.command_name
    )
    if args.action == "start":
        desired_runid = (
            args.desired_runid
            if args.desired_runid is not None
            else int(current_runid) + 1
        )
        force_quit = 0
    else:
        # SheriffDeputyCommand._stop() sets force_quit=1; desired_runid stays at the
        # running instance's actual_runid.
        desired_runid = (
            args.desired_runid
            if args.desired_runid is not None
            else int(current_runid)
        )
        force_quit = 1

    if args.replace_all:
        msg = _build_orders_msg(
            host=args.host,
            sheriff_name=args.sheriff_name,
            sheriff_id=args.sheriff_id,
            group=args.group,
            command_name=args.command_name,
            exec_str=args.exec_str,
            desired_runid=desired_runid,
            force_quit=force_quit,
        )
    else:
        msg = _build_merged_orders_msg(
            info=info,
            host=args.host,
            sheriff_name=args.sheriff_name,
            sheriff_id=args.sheriff_id,
            group=args.group,
            command_name=args.command_name,
            exec_str=args.exec_str,
            action=args.action,
            desired_runid_override=args.desired_runid,
        )
    lc.publish(args.orders_channel, msg.encode())

    extra = ""
    if not args.replace_all:
        extra = " ncmds={} (merged from PMD_INFO2)".format(msg.ncmds)

    print(
        "Published {} order: host='{}' command='{}:{}' actual_runid={} desired_runid={} force_quit={} channel={}{}".format(
            args.action,
            args.host,
            args.group,
            args.command_name,
            current_runid,
            desired_runid,
            force_quit,
            args.orders_channel,
            extra,
        )
    )


if __name__ == "__main__":
    main()

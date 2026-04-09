import argparse
import shlex
import subprocess


class ControllerCommandSenser:
    def __init__(
        self,
        remote_host: str = "hienbui@158.130.52.50",
        remote_workdir: str = "/home/hienbui/git/dairlib",
        remote_exec: str = (
            "bazel-bin/examples/sampling_c3/franka_sampling_c3_controller "
            "--is_simulation=true --demo_name=anything"
        ),
        remote_pid_file: str = "/tmp/franka_sampling_c3_controller.pid",
        remote_log_file: str = "/tmp/franka_sampling_c3_controller.log",
        ssh_connect_timeout_sec: int = 5,
        ssh_command_timeout_sec: int = 10,
    ) -> None:
        self.remote_host = remote_host
        self.remote_workdir = remote_workdir
        self.remote_exec = remote_exec
        self.remote_pid_file = remote_pid_file
        self.remote_log_file = remote_log_file
        self.ssh_connect_timeout_sec = ssh_connect_timeout_sec
        self.ssh_command_timeout_sec = ssh_command_timeout_sec
        self.state = "stopped"

    def _run_ssh(self, remote_cmd: str, check: bool = True) -> None:
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-n",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    f"ConnectTimeout={self.ssh_connect_timeout_sec}",
                    "-o",
                    "ConnectionAttempts=1",
                    self.remote_host,
                    remote_cmd,
                ],
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=self.ssh_command_timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "SSH command timed out after "
                f"{self.ssh_command_timeout_sec}s. "
                "The remote command may still be running."
            ) from exc
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )

    def _spawn_ssh_detached(self, remote_cmd: str) -> None:
        subprocess.Popen(
            [
                "ssh",
                "-n",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={self.ssh_connect_timeout_sec}",
                "-o",
                "ConnectionAttempts=1",
                self.remote_host,
                remote_cmd,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    def start_remote(self) -> None:
        remote_cmd = (
            f"cd {shlex.quote(self.remote_workdir)} && "
            f"setsid {self.remote_exec} < /dev/null > {shlex.quote(self.remote_log_file)} 2>&1 & "
            f"echo $! > {shlex.quote(self.remote_pid_file)}"
        )
        self._spawn_ssh_detached(remote_cmd)
        self.state = "running"
        print("Start command dispatched in background.")

    def stop_remote(self) -> None:
        exec_binary = self.remote_exec.split()[0]
        exec_name = exec_binary.split("/")[-1]
        remote_cmd = (
            f"if [ -f {shlex.quote(self.remote_pid_file)} ]; then "
            f"PID=$(cat {shlex.quote(self.remote_pid_file)}); "
            "if kill -0 $PID >/dev/null 2>&1; then "
            "kill $PID; "
            "sleep 1; "
            "if kill -0 $PID >/dev/null 2>&1; then "
            "kill -9 $PID; "
            "fi; "
            "echo Stopped PID $PID; "
            "else "
            "echo PID file exists but process is not running; "
            "fi; "
            f"rm -f {shlex.quote(self.remote_pid_file)}; "
            "else "
            "echo PID file not found; "
            "fi; "
            f"pkill -9 -f {shlex.quote(exec_name)} >/dev/null 2>&1 || true; "
            f"killall -9 {shlex.quote(exec_name)} >/dev/null 2>&1 || true; "
            f"if pgrep -af {shlex.quote(exec_name)} >/dev/null 2>&1; then "
            f"echo WARNING: matching process still running for {shlex.quote(exec_name)}; "
            f"pgrep -af {shlex.quote(exec_name)}; "
            "else "
            "echo No matching process remains; "
            "fi"
        )
        self._run_ssh(remote_cmd, check=False)
        self.state = "stopped"

    def is_running(self) -> bool:
        return self.state == "running"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start/stop remote executable via ssh without blocking."
    )
    parser.add_argument(
        "--action",
        choices=("start", "stop"),
        default="start",
        help="start runs executable in background, stop kills it by PID file.",
    )
    args = parser.parse_args()
    controller = ControllerCommandSenser()

    if args.action == "start":
        controller.start_remote()
    else:
        controller.stop_remote()


if __name__ == "__main__":
    main()

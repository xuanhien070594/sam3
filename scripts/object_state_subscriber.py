import lcm
from lcmtypes.lcmt_object_state.lcmt_object_state import lcmt_object_state
from loguru import logger


class ObjectStateSubscriber:
    def __init__(self, channel: str = "OBJECT_STATE") -> None:
        self.channel = channel
        self.has_received = False
        self.lc = lcm.LCM()
        self.lc.subscribe(self.channel, self._on_object_state)

    def _on_object_state(self, channel: str, data: bytes) -> None:
        if self.has_received:
            return

        _ = lcmt_object_state.decode(data)

        self.has_received = True
        logger.info("[{}] message received", channel)

    def poll(self, timeout_ms: int = 0) -> bool:
        status = self.lc.handle_timeout(timeout_ms)
        return status == 1

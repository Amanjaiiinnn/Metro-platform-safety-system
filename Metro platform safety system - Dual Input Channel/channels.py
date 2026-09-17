"""
channels.py
-----------
SingleSlotChannel: a 1-item mailbox between two threads.

put() always overwrites whatever's currently sitting there, unread. That
is the "drop the pending old job, replace it with the newest" policy used
everywhere in the threaded pipeline (frame handoff, NPU job submission,
NPU results) -- for a real-time tripwire, the freshest frame is worth more
than processing every single one, so a slow consumer skips stale work
instead of building a backlog.
"""

import threading


class SingleSlotChannel:
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._item = None
        self._has_item = False

    def put(self, item):
        """Overwrites any unread item currently in the slot."""
        with self._cond:
            self._item = item
            self._has_item = True
            self._cond.notify()

    def get(self, timeout=None):
        """Blocks until an item is available or timeout elapses.
        Returns the item, or None on timeout. None is also used in this
        codebase as an explicit EOF/stop signal -- callers distinguish the
        two by context (e.g. "was this a live source?").
        """
        with self._cond:
            ready = self._cond.wait_for(lambda: self._has_item, timeout=timeout)
            if not ready:
                return None
            item = self._item
            self._item = None
            self._has_item = False
            return item

import threading
import sys
import time


class CLIProgress:
    def __init__(self):
        self.chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_file = ""
        self.running = False
        self._paused = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def set_file(self, filename: str):
        with self._lock:
            self.current_file = filename

    def pause_and_clear(self):
        self._paused = True

        with self._lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def resume(self):
        self._paused = False

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _animate(self):
        i = 0
        while self.running:
            if not self._paused:
                with self._lock:
                    if self.current_file:
                        sys.stdout.write(
                            f"\r{self.chars[i % len(self.chars)]} {self.current_file}\033[K"
                        )
                        sys.stdout.flush()
                i += 1
            time.sleep(0.08)

"""Run the full FYP-Pulse pipeline once: clean -> train/select -> tiers -> fairness -> artifacts/."""
import time

import core

if __name__ == "__main__":
    t = time.time()
    res = core.build(verbose=True)
    print(f"\nBuild finished in {time.time() - t:.1f}s")

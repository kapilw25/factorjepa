"""Zero-idle decode feeder — the DataLoader prefetch pattern as ONE shared util (iter19 2026-07-09).

Gold standard: torch DataLoader's worker+prefetch design ("with prefetch the GPU never waits
because batches are always ready" — https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html,
https://www.jpatrickpark.com/post/prefetcher/); in-repo template: utils/frozen_features.py's
decode-worker pool (measured 99% GPU on the E: stage).

Why this exists (measured on FULL, 2026-07-08): m12d decoded clips SERIALLY inside its forward
loop; m12e/m12f used a submit-then-refill ThreadPoolExecutor whose refill only ran on the MAIN
thread between future-pops — during a 32-clip GPU forward (~16s) nothing refilled, the workers
drained their in-flight futures (~4s) then sat idle, and the GPU square-waved (pace: 17-25% util).
HERE the N worker threads own clip_q → decode → ready_q end-to-end and never pause, whatever the
consumer is doing; the bounded ready_q caps RAM (depth × per-clip tensor bytes) and always holds
the next batches ready when a forward ends. Every m12 eval module shares THIS class — no more
per-module divergent decode loops (the m12e-got-the-pool/m12f-didn't asymmetry class).

USAGE (inside an m12 eval module — n_workers/depth come from the module's pipeline.yaml keys):
    from utils.decode_feeder import DecodeFeeder
    clip_q, tar_stop, _r = iter_clips_parallel(local_data=..., subset_keys=...)
    feeder = DecodeFeeder(
        clip_q,
        decode_one=lambda ck, mp4: decode_to_tensor(mp4, tmp_dir, ck, num_frames, crop),
        n_workers=args.decode_workers, ready_depth=args.batch_size * 3, timeout_s=300)
    for clip_key, tensor in feeder:      # decode failures are skipped + logged inside
        ...batch → GPU forward...        # decode NEVER pauses while this runs
"""
import queue
import threading

import torch


class DecodeFeeder:
    """N decode threads: clip_q → decode_one(ck, mp4) → bounded ready_q; iterate to consume.

    End-of-stream: a worker seeing clip_q's None sentinel re-broadcasts it for its siblings
    (frozen_features idiom) and exits; a joiner thread puts the ready_q sentinel once every
    worker has drained — so the consumer's for-loop simply ends. A clip whose decode returns
    None or raises is logged and skipped (per-clip fail-soft, matching the serial paths).
    """

    def __init__(self, clip_q, decode_one, n_workers, ready_depth, timeout_s):
        if n_workers < 1 or ready_depth < 1:
            raise ValueError(f"DecodeFeeder: n_workers={n_workers} ready_depth={ready_depth} must be ≥1")
        self._ready: "queue.Queue" = queue.Queue(maxsize=ready_depth)
        torch.set_num_threads(1)   # decode_one's torch resize runs 1-thread per worker → the N
        #                            workers parallelize cleanly instead of fighting over all cores

        def _worker():
            while True:
                try:
                    item = clip_q.get(timeout=timeout_s)
                except queue.Empty:
                    print("  WARN: DecodeFeeder clip-queue timeout — worker draining out")
                    break
                if item is None:
                    clip_q.put(None)            # re-broadcast sentinel to sibling workers
                    break
                ck, mp4 = item
                try:
                    t = decode_one(ck, mp4)
                except Exception as e:          # per-clip FAIL-SOFT (matches the serial paths' None-skip)
                    print(f"  SKIP (decode error {ck}): {e}")
                    t = None
                if t is None:
                    print(f"  SKIP (decode fail): {ck}")
                    continue
                self._ready.put((ck, t))

        self._workers = [threading.Thread(target=_worker, daemon=True) for _ in range(n_workers)]
        for w in self._workers:
            w.start()

        def _joiner():                          # ready_q sentinel once ALL workers drained
            for w in self._workers:
                w.join()
            self._ready.put(None)

        threading.Thread(target=_joiner, daemon=True).start()

    def __iter__(self):
        while True:
            item = self._ready.get()
            if item is None:
                return
            yield item


class BatchFeeder:
    """Second pipeline stage (iter19 2026-07-09, zero-idle rung 2): a builder thread pops decoded
    clips from a DecodeFeeder, torch.stack's them into forward-ready (keys, batch) pairs and
    optionally pins the batch, publishing into a bounded batch_q. depth=2 = classic double
    buffering: batch k+1 is ASSEMBLED (the 1-2s single-thread stack of GBs that used to run on
    the GPU loop between forwards) while batch k forwards, and a pinned batch makes the
    downstream .to("cuda") a straight DMA copy. Values are IDENTICAL — same torch.stack, same
    clip order, no arithmetic. The tail partial batch is published as-is; iteration ends when
    the upstream feeder ends.

    USAGE:
        batches = BatchFeeder(feeder, batch_size=args.batch_size, depth=2, pin_memory=True)
        for keys, batch_t in batches:          # batch_t: (B, T, 3, H, W) fp32, pinned
            ...GPU forward...                  # assembly of the NEXT batch overlaps this
    """

    def __init__(self, feeder, batch_size, depth, pin_memory):
        if batch_size < 1 or depth < 1:
            raise ValueError(f"BatchFeeder: batch_size={batch_size} depth={depth} must be ≥1")
        self._q: "queue.Queue" = queue.Queue(maxsize=depth)

        def _pack(keys, tens):
            batch = torch.stack(tens)
            if pin_memory:
                batch = batch.pin_memory()
            return keys, batch

        def _builder():
            keys, tens = [], []
            for ck, t in feeder:
                keys.append(ck)
                tens.append(t)
                if len(tens) >= batch_size:
                    self._q.put(_pack(keys, tens))
                    keys, tens = [], []
            if tens:
                self._q.put(_pack(keys, tens))
            self._q.put(None)

        threading.Thread(target=_builder, daemon=True).start()

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

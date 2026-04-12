/*
 * mmap.bpf.c — BPF probe for anonymous memory mapping growth.
 *
 * Kernel requirement: Linux ≥ 4.9, CAP_BPF + CAP_PERFMON (or CAP_SYS_ADMIN)
 *
 * Tracepoints attached:
 *   syscalls:sys_enter_mmap  — fires on every mmap() syscall.
 *                              Only anonymous, private mappings are reported
 *                              (MAP_ANONYMOUS | MAP_PRIVATE).
 *   syscalls:sys_enter_brk   — fires on every brk() syscall.
 *                              Only expansions (new_brk > prev_brk) are
 *                              reported; queries (brk(0)) are dropped.
 *
 * Use case — pre-RSS expansion detection:
 *   When a vLLM / SGLang process commits anonymous memory via mmap/brk,
 *   the pages are allocated but not yet faulted in — the commitment is
 *   invisible to /proc/meminfo (RSS) and to vLLM's Prometheus metrics until
 *   the pages are actually touched.  This probe catches the commitment
 *   immediately, reporting mmap_growth_mbps before any poll-based approach.
 *
 * Wire format — struct mmap_event
 * (must match _MmapEvent in mmap_growth.py exactly):
 *
 *   offset  size  field
 *   ------  ----  ------
 *     0      8    timestamp_ns    — bpf_ktime_get_ns() at event time
 *     8      8    alloc_bytes     — bytes committed (mmap len or brk delta)
 *    16      4    pid             — process PID
 *    20      4    event_subtype   — 0 = mmap, 1 = brk
 *
 * Total: 24 bytes per event.
 *
 * PID allowlist: same design as page_fault.bpf.c — see header comment there.
 *
 * This file is compiled at runtime by the bcc Python library (BPF(text=...)).
 */

#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

#define SUBTYPE_MMAP 0
#define SUBTYPE_BRK  1

/*
 * MAP_ANONYMOUS (0x20) and MAP_PRIVATE (0x02) flags — x86-64 Linux ABI.
 * We only report anonymous, private mappings; skip file-backed or shared maps.
 */
#define MAP_ANONYMOUS_FLAG 0x20
#define MAP_PRIVATE_FLAG   0x02

struct mmap_event {
    u64  timestamp_ns;
    u64  alloc_bytes;
    u32  pid;
    u32  event_subtype;
};

BPF_PERF_OUTPUT(mmap_events);

BPF_HASH(pid_allowlist_mmap, u32, u8);
BPF_ARRAY(pid_filter_count_mmap, u32, 1);

/*
 * brk_state: per-PID last-known brk address.
 * Lets us compute the growth delta on each brk() call.
 */
BPF_HASH(brk_state, u32, u64);

/* ---- PID filter helper ---- */
static __always_inline int _pid_allowed(u32 pid) {
    u32 zero = 0;
    u32 *cnt = pid_filter_count_mmap.lookup(&zero);
    if (!cnt || *cnt == 0) return 1;   /* empty allowlist → pass all */
    u8 *ok = pid_allowlist_mmap.lookup(&pid);
    return ok ? 1 : 0;
}

/* ---- syscalls:sys_enter_mmap ---- */
TRACEPOINT_PROBE(syscalls, sys_enter_mmap) {
    u32 pid = (u32)(bpf_get_current_pid_tgid() >> 32);
    if (!_pid_allowed(pid)) return 0;

    /* Only anonymous, private mappings */
    unsigned long flags = args->flags;
    if (!(flags & MAP_ANONYMOUS_FLAG)) return 0;
    if (!(flags & MAP_PRIVATE_FLAG))   return 0;

    u64 len = (u64)args->len;
    if (len == 0) return 0;

    struct mmap_event ev = {};
    ev.timestamp_ns  = bpf_ktime_get_ns();
    ev.alloc_bytes   = len;
    ev.pid           = pid;
    ev.event_subtype = SUBTYPE_MMAP;

    mmap_events.perf_submit(ctx, &ev, sizeof(ev));
    return 0;
}

/* ---- syscalls:sys_enter_brk ---- */
TRACEPOINT_PROBE(syscalls, sys_enter_brk) {
    u32 pid = (u32)(bpf_get_current_pid_tgid() >> 32);
    if (!_pid_allowed(pid)) return 0;

    u64 new_brk = (u64)args->brk;
    if (new_brk == 0) return 0;   /* brk(0) = query current address, skip */

    u64 *prev = brk_state.lookup(&pid);
    u64 growth = 0;
    if (prev && *prev > 0 && new_brk > *prev) {
        growth = new_brk - *prev;
    }

    /* Always update state so next call has the correct baseline */
    brk_state.update(&pid, &new_brk);

    if (growth == 0) return 0;

    struct mmap_event ev = {};
    ev.timestamp_ns  = bpf_ktime_get_ns();
    ev.alloc_bytes   = growth;
    ev.pid           = pid;
    ev.event_subtype = SUBTYPE_BRK;

    mmap_events.perf_submit(ctx, &ev, sizeof(ev));
    return 0;
}

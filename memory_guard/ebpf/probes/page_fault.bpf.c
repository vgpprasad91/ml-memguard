/*
 * page_fault.bpf.c — BPF probe for user-space page fault monitoring.
 *
 * Kernel requirement: Linux ≥ 4.9, CAP_BPF + CAP_PERFMON (or CAP_SYS_ADMIN)
 *
 * Tracepoint attached:
 *   exceptions:page_fault_user
 *       Fires on every user-space page fault, including:
 *         - Minor (soft) faults  — page present in page cache, no disk I/O
 *         - Major (hard) faults  — page must be read from swap or disk
 *
 * Use case — "silent kill" detection:
 *   When a GPU-serving process (vLLM, SGLang) exhausts GPU memory and the
 *   kernel begins reclaiming pages, the process starts faulting into swap.
 *   Page-fault rate spikes 100-1000x above baseline in the 300–800 ms window
 *   before the OOM killer fires SIGKILL.  Because SIGKILL bypasses Python's
 *   signal machinery, the current VLLMWatchdog misses the kill entirely and
 *   only notices the dead process on the next poll tick (up to 10 seconds
 *   later).  This probe delivers the signal in microseconds.
 *
 * Wire format — struct page_fault_event
 * (must match _PageFaultEvent in page_fault.py exactly):
 *
 *   offset  size  field
 *   ------  ----  ------
 *     0      8    timestamp_ns   — bpf_ktime_get_ns() at event time
 *     8      8    fault_address  — faulting virtual address
 *    16      4    error_code     — page fault error code bits
 *    20      4    pid            — triggering process PID
 *
 * Total: 24 bytes per event.
 *
 * PID allowlist:
 *   pid_allowlist  — BPF_HASH(u32 pid → u8 1).  Python manages entries
 *                    via probe.add_pid() / probe.remove_pid().
 *   pid_filter_count — BPF_ARRAY index 0 holds the count of PIDs in
 *                    pid_allowlist.  When 0, all PIDs pass through.
 *
 * This file is compiled at runtime by the bcc Python library (BPF(text=...)).
 * Do not build it with clang -target bpf directly.
 */

#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct page_fault_event {
    u64  timestamp_ns;
    u64  fault_address;
    u32  error_code;
    u32  pid;
};

BPF_PERF_OUTPUT(page_fault_events);

/*
 * pid_allowlist: key=PID, value=1.  Populated by probe.add_pid() calls.
 * Empty = watch all PIDs (no filter active).
 */
BPF_HASH(pid_allowlist, u32, u8);

/*
 * pid_filter_count[0] = number of entries in pid_allowlist.
 * Maintained by the Python wrapper: incremented on add_pid(), decremented
 * on remove_pid().  Checking a counter is faster than testing map size.
 */
BPF_ARRAY(pid_filter_count, u32, 1);

TRACEPOINT_PROBE(exceptions, page_fault_user) {
    u32 pid = (u32)(bpf_get_current_pid_tgid() >> 32);

    /* ---- PID filter ---- */
    u32 zero = 0;
    u32 *cnt = pid_filter_count.lookup(&zero);
    if (cnt && *cnt > 0) {
        u8 *allowed = pid_allowlist.lookup(&pid);
        if (!allowed) return 0;   /* not in allowlist — drop event */
    }

    struct page_fault_event ev = {};
    ev.timestamp_ns  = bpf_ktime_get_ns();
    ev.fault_address = args->address;
    ev.error_code    = (u32)args->error_code;
    ev.pid           = pid;

    page_fault_events.perf_submit(ctx, &ev, sizeof(ev));
    return 0;
}

/*
 * cgroup_memory.bpf.c — BPF probe for cgroup memory.high pressure events.
 *
 * Kernel requirement:  Linux ≥ 5.8, cgroup v2, CAP_BPF + CAP_PERFMON
 *                      (or CAP_SYS_ADMIN on kernels < 5.8)
 *
 * Tracepoint attached:
 *   cgroup:cgroup_memory_high
 *       Fires when a cgroup's memory.usage_in_bytes exceeds its
 *       memory.high soft limit.  This is 200–500 ms before the kernel
 *       OOM killer selects a victim — it is the earliest user-visible
 *       signal that the cgroup is in trouble.
 *
 *       Tracepoint args
 *       (see /sys/kernel/debug/tracing/events/cgroup/cgroup_memory_high/format):
 *         args->path    — null-terminated cgroup path string (e.g. /kubepods/pod-abc/…)
 *         args->actual  — current memory.usage_in_bytes
 *         args->high    — memory.high limit in bytes
 *
 * Wire format — struct cgroup_mem_high_event (must match _CgroupMemHighEvent in Python):
 *
 *   offset  size  field
 *   ------  ----  ------
 *      0      8   timestamp_ns    — bpf_ktime_get_ns() at event time (monotonic)
 *      8      8   pressure_bytes  — actual - high (bytes over the high watermark)
 *     16      4   pid             — triggering process PID
 *     20      4   _pad            — alignment padding (zero-filled)
 *     24    128   cgroup_id       — null-terminated cgroup path (128 bytes max)
 *
 * Total: 152 bytes per event.
 *
 * This file is compiled at runtime by the bcc (BPF Compiler Collection)
 * Python library via BPF(text=...).  It is NOT a standalone eBPF object
 * file — do not build it with clang -target bpf directly.
 */

#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

#define CGROUP_PATH_LEN 128

/* ---- Event wire format ---- */
/* Must stay in sync with _CgroupMemHighEvent in cgroup_memory_high.py */
struct cgroup_mem_high_event {
    u64  timestamp_ns;
    u64  pressure_bytes;
    u32  pid;
    u32  _pad;                     /* alignment — always zero */
    char cgroup_id[CGROUP_PATH_LEN];
};

BPF_PERF_OUTPUT(cgroup_mem_high_events);

/*
 * cgroup:cgroup_memory_high
 * -------------------------
 * Fires inside the kernel's memory reclaim path when a cgroup's
 * memory.usage_in_bytes exceeds memory.high.  Because this fires
 * synchronously in the allocating task's context it precedes any
 * userspace-visible side-effect by hundreds of milliseconds — no
 * /proc/meminfo read or Prometheus scrape can deliver this signal
 * as early.
 *
 * pressure_bytes computation:
 *   If actual > high, pressure_bytes = actual - high.
 *   If actual <= high (e.g. cgroup reclaim already succeeded by the
 *   time we run), pressure_bytes = 0.  The event is still delivered
 *   so the Python layer can count crossings even without pressure delta.
 */
TRACEPOINT_PROBE(cgroup, cgroup_memory_high) {
    struct cgroup_mem_high_event ev = {};
    ev.timestamp_ns = bpf_ktime_get_ns();
    ev.pid          = (u32)(bpf_get_current_pid_tgid() >> 32);

    u64 actual = (u64)args->actual;
    u64 high   = (u64)args->high;
    ev.pressure_bytes = (actual > high) ? (actual - high) : 0;

    /*
     * bpf_probe_read_kernel_str is safe even if args->path is a
     * dynamically allocated __data_loc string — the BPF verifier
     * validates the bounds.  Returns number of bytes copied (incl. NUL).
     */
    bpf_probe_read_kernel_str(
        ev.cgroup_id, sizeof(ev.cgroup_id),
        (const void *)args->path
    );

    cgroup_mem_high_events.perf_submit(ctx, &ev, sizeof(ev));
    return 0;
}

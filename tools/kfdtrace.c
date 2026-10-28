/*
 * kfdtrace.c - log what a process asks the amdgpu KFD driver to allocate.
 *
 * Build:  gcc -O2 -shared -fPIC -o libkfdtrace.so kfdtrace.c -ldl -lpthread
 * Use:    LD_PRELOAD=/path/libkfdtrace.so KFDTRACE_DIR=/some/dir <program>
 *
 * Wraps ioctl() and records the KFD calls that the driver charges to its
 * system-memory limit (kfd_mem_limit in amdgpu_amdkfd_gpuvm.c):
 *   ALLOC_MEMORY_OF_GPU (0x16)  with its size and flags; USERPTR and GTT
 *                               count against the system limit, VRAM does not
 *   FREE_MEMORY_OF_GPU  (0x17)  the matching release, looked up by handle
 *   SVM                 (0x20)  range and op, for reference
 * A USERPTR range that lies inside a mapping of a file named *.arena is
 * tagged "arena", so the expert arena can be told apart from the rest.
 *
 * Output: one line per call in KFDTRACE_DIR/kfd-<pid>.tsv:
 *   t_s  op  tag  size  flags  ret  sys_now  sys_peak  arena_now  addr
 * where sys_now is the running sum of USERPTR + GTT bytes this process holds.
 * SVM ranges are charged by the driver when they are created (kfd_svm.c,
 * svm_range_new, when XNACK is off); repeated set_attr calls on a range that
 * already exists are not charged again, so the SVM charge is the union of the
 * logged [addr, addr+size) ranges, worked out afterwards.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define F_VRAM (1u << 0)
#define F_GTT (1u << 1)
#define F_USERPTR (1u << 2)

struct alloc_args {
    uint64_t va_addr, size, handle, mmap_offset;
    uint32_t gpu_id, flags;
};
struct free_args {
    uint64_t handle;
};
struct svm_args {
    uint64_t start_addr, size;
    uint32_t op, nattr;
};

#define MAXH 65536
static struct {
    uint64_t handle, size;
    uint32_t flags;
    int arena;
} live[MAXH];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int (*real_ioctl)(int, unsigned long, ...);
static int out = -1;
static int64_t sys_now, sys_peak, arena_now;
static double t0;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void open_out(void) {
    const char* dir = getenv("KFDTRACE_DIR");
    char path[512];
    snprintf(path, sizeof path, "%s/kfd-%d.tsv", dir ? dir : "/tmp", (int) getpid());
    out = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    t0 = now_s();
    if (out >= 0) {
        char exe[256] = "";
        ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n > 0) exe[n] = 0;
        dprintf(out, "# exe %s\n# t_s\top\ttag\tsize\tflags\tret\tsys_now\tsys_peak\tarena_now\n", exe);
    }
}

/* 1 when [va, va+size) lies inside a mapping whose path ends in ".arena" */
static int in_arena(uint64_t va, uint64_t size) {
    FILE* f = fopen("/proc/self/maps", "re");
    if (!f) return 0;
    char line[1024];
    int hit = 0;
    while (!hit && fgets(line, sizeof line, f)) {
        unsigned long lo, hi;
        if (sscanf(line, "%lx-%lx", &lo, &hi) != 2) continue;
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == ' ')) line[--L] = 0;
        if (L > 6 && strcmp(line + L - 6, ".arena") == 0 && va >= lo && va + size <= hi) hit = 1;
    }
    fclose(f);
    return hit;
}

static void logline_at(const char* op, const char* tag, uint64_t addr, uint64_t size, uint32_t flags, int ret) {
    if (out < 0) return;
    dprintf(out, "%.3f\t%s\t%s\t%llu\t0x%x\t%d\t%lld\t%lld\t%lld\t0x%llx\n", now_s() - t0, op, tag,
            (unsigned long long) size, flags, ret, (long long) sys_now, (long long) sys_peak, (long long) arena_now,
            (unsigned long long) addr);
}

static void logline(const char* op, const char* tag, uint64_t size, uint32_t flags, int ret) {
    logline_at(op, tag, 0, size, flags, ret);
}

int ioctl(int fd, unsigned long req, ...) {
    va_list ap;
    va_start(ap, req);
    void* arg = va_arg(ap, void*);
    va_end(ap);
    if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, ...)) dlsym(RTLD_NEXT, "ioctl");
    if (_IOC_TYPE(req) != 'K') return real_ioctl(fd, req, arg);

    unsigned nr = _IOC_NR(req);
    if (nr == 0x16) {
        struct alloc_args* a = arg;
        uint64_t size = a->size;
        uint32_t flags = a->flags;
        int arena = (flags & F_USERPTR) && in_arena(a->va_addr, size);
        int ret = real_ioctl(fd, req, arg);
        pthread_mutex_lock(&mu);
        if (out < 0) open_out();
        int sys = (flags & (F_USERPTR | F_GTT)) != 0;
        if (ret == 0) {
            if (sys) sys_now += size;
            if (arena) arena_now += size;
            if (sys_now > sys_peak) sys_peak = sys_now;
            for (int i = 0; i < MAXH; ++i)
                if (live[i].handle == 0) {
                    live[i].handle = a->handle;
                    live[i].size = size;
                    live[i].flags = flags;
                    live[i].arena = arena;
                    break;
                }
        }
        const char* tag = arena ? "arena" : (flags & F_USERPTR) ? "userptr" : (flags & F_GTT) ? "gtt"
                                           : (flags & F_VRAM) ? "vram" : "other";
        logline_at(ret == 0 ? "alloc" : "alloc-FAIL", tag, a->va_addr, size, flags, ret);
        pthread_mutex_unlock(&mu);
        return ret;
    }
    if (nr == 0x17) {
        uint64_t h = ((struct free_args*) arg)->handle;
        int ret = real_ioctl(fd, req, arg);
        pthread_mutex_lock(&mu);
        if (out < 0) open_out();
        for (int i = 0; i < MAXH; ++i)
            if (live[i].handle == h) {
                if (ret == 0) {
                    if (live[i].flags & (F_USERPTR | F_GTT)) sys_now -= live[i].size;
                    if (live[i].arena) arena_now -= live[i].size;
                }
                logline(ret == 0 ? "free" : "free-FAIL", live[i].arena ? "arena" : "-", live[i].size, live[i].flags, ret);
                live[i].handle = 0;
                break;
            }
        pthread_mutex_unlock(&mu);
        return ret;
    }
    if (nr == 0x20) {
        struct svm_args* s = arg;
        uint64_t start = s->start_addr, size = s->size;
        uint32_t op = s->op;
        int arena = in_arena(start, size);
        int ret = real_ioctl(fd, req, arg);
        pthread_mutex_lock(&mu);
        if (out < 0) open_out();
        logline_at(ret == 0 ? "svm" : "svm-FAIL", op != 0 ? "get_attr" : arena ? "set_attr-arena" : "set_attr",
                   start, size, op, ret);
        /* who registers small pieces of the arena: the first few call stacks (KFDTRACE_BT=N) */
        static int bt_left = -1;
        if (bt_left < 0) bt_left = getenv("KFDTRACE_BT") ? atoi(getenv("KFDTRACE_BT")) : 0;
        if (arena && size < (64u << 20) && bt_left > 0 && out >= 0) {
            --bt_left;
            void* frames[40];
            int n = backtrace(frames, 40);
            dprintf(out, "# backtrace of the svm set_attr above\n");
            backtrace_symbols_fd(frames, n, out);
        }
        pthread_mutex_unlock(&mu);
        return ret;
    }
    return real_ioctl(fd, req, arg);
}

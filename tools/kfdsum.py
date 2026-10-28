"""Summarize a kfdtrace log: what one process was charged against the amdgpu KFD system-memory limit.

Usage: python3 kfdsum.py kfd-PID.tsv

kfdtrace.c writes one line per KFD call.  The driver charges an SVM range once, when the range is created, so the
SVM charge is the union of the logged [addr, addr+size) ranges.  USERPTR and GTT allocations are charged by size until
they are freed.  "arena" is the shared expert arena (a mapping of a file named *.arena); "copies" are small arena
ranges registered after the arena itself, which is what a copy from an unregistered arena page leaves behind.
"""
import sys

GiB = 2 ** 30
COPY_MAX = 64 << 20   # an arena range smaller than this is a copy's registration, not a slice of the arena


def rows(path):
    """Data lines of the log as (t, op, tag, size, ret, addr); comment and call-stack lines are skipped."""
    out = []
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        if line.startswith('#') or len(f) < 10:
            continue
        try:
            out.append((float(f[0]), f[1], f[2], int(f[3]), int(f[5]), int(f[9], 16)))
        except ValueError:
            continue
    return out


def union(ranges):
    """Bytes covered by a list of [start, end) ranges."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(ranges):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def main():
    r = rows(sys.argv[1])
    svm = [(t, tag, a, a + sz) for t, op, tag, sz, ret, a in r if op == 'svm' and tag.startswith('set_attr')]
    arena = [(s, e) for t, tag, s, e in svm if tag == 'set_attr-arena' and e - s >= COPY_MAX]
    copies = [(s, e) for t, tag, s, e in svm if tag == 'set_attr-arena' and e - s < COPY_MAX]
    other = [(s, e) for t, tag, s, e in svm if tag == 'set_attr']
    held = 0   # kfdtrace's running sum of USERPTR + GTT bytes, as of the last alloc or free in the log
    for line in open(sys.argv[1]):
        f = line.split('\t')
        if not line.startswith('#') and len(f) >= 10 and f[1] in ('alloc', 'free'):
            held = int(f[6])
    refused = sum(1 for x in r if x[1].endswith('FAIL'))
    print(f'arena registered        {union(arena) / GiB:7.2f} GiB')
    print(f'arena pages by copies   {union(copies) / GiB:7.2f} GiB  ({len(copies)} calls)')
    print(f'other SVM ranges        {union(other) / GiB:7.2f} GiB')
    print(f'GTT and userptr held    {held / GiB:7.2f} GiB')
    print(f'refused calls           {refused}')


if __name__ == '__main__':
    main()

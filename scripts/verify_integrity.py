#!/usr/bin/env python3
"""Check files listed in the unsigned release SHA256SUMS manifest; stdlib only."""
import argparse
import hashlib
import re
import sys
from pathlib import Path, PurePosixPath


def verify(root):
    root = root.resolve()
    lines = (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    seen, failures = set(), []
    for number, line in enumerate(lines, 1):
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match:
            raise ValueError(f'malformed manifest line {number}')
        expected, name = match.groups()
        relative = PurePosixPath(name)
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or ':' in name or name in seen or relative.as_posix() != name):
            raise ValueError(f'invalid or repeated manifest path on line {number}')
        seen.add(name)
        target = root.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(root):
            raise ValueError(f'path outside repository on line {number}')
        if target.is_symlink() or not target.is_file():
            failures.append(f'missing or nonregular: {name}')
        elif hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            failures.append(f'changed: {name}')
    if not seen:
        raise ValueError('empty manifest')
    return len(seen), failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    try:
        count, failures = verify(args.root)
    except (OSError, ValueError) as error:
        print(f'Integrity check failed: {error}', file=sys.stderr)
        return 2
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print(f'Integrity verified for {count} listed files. Manifest is unsigned.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

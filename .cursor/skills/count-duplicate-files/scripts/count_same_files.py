#!/usr/bin/env python3
"""
递归统计目录下「相同文件」出现次数，并按次数从多到少排列。
默认递归当前目录。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "build", "dist", "_skbuild", ".mypy_cache", ".ruff_cache",
}


def file_content_hash(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def collect_by_basename(root: Path, ignore_dirs: set[str], follow_symlinks: bool) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirpath = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for name in filenames:
            full = dirpath / name
            if not full.is_file():
                continue
            try:
                groups.setdefault(full.name, []).append(full)
            except Exception:
                pass
    return groups


def collect_by_content(root: Path, ignore_dirs: set[str], follow_symlinks: bool) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirpath = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for name in filenames:
            full = dirpath / name
            if not full.is_file():
                continue
            try:
                h = file_content_hash(full)
                groups.setdefault(h, []).append(full)
            except (OSError, PermissionError):
                pass
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="递归统计目录下相同文件出现次数，按次数从多到少排列。默认递归当前目录。")
    parser.add_argument("dir", nargs="?", default=".", type=Path, help="要扫描的根目录（默认当前目录）")
    parser.add_argument("--by-content", action="store_true", help="按文件内容哈希统计")
    parser.add_argument("--no-ignore", action="store_true", help="不忽略 .git、__pycache__ 等目录")
    parser.add_argument("--follow-symlinks", action="store_true", help="跟随符号链接")
    parser.add_argument("--top", type=int, default=0, metavar="N", help="只显示前 N 个（0=全部）")
    parser.add_argument("--min-count", type=int, default=2, metavar="N", help="只显示出现次数>=N的项（默认2）")
    parser.add_argument("--show-paths", action="store_true", help="每行下列出所有路径")
    args = parser.parse_args()

    root = args.dir.resolve()
    if not root.is_dir():
        print(f"错误：不是目录或不存在: {root}", file=sys.stderr)
        return 1

    ignore_dirs = set() if args.no_ignore else DEFAULT_IGNORE_DIRS

    if args.by_content:
        groups = collect_by_content(root, ignore_dirs, args.follow_symlinks)
        key_label = "content_hash"
    else:
        groups = collect_by_basename(root, ignore_dirs, args.follow_symlinks)
        key_label = "filename"

    total_files = sum(len(paths) for paths in groups.values())

    items = [(k, paths) for k, paths in groups.items() if len(paths) >= args.min_count]
    items.sort(key=lambda x: -len(x[1]))
    if args.top > 0:
        items = items[: args.top]

    print(f"# 扫描根目录（递归）: {root}")
    print(f"# 总文件数目: {total_files}")
    print(f"# 按 {key_label} 统计，出现次数 >= {args.min_count}，按次数降序")
    print()

    if not items:
        print("没有满足条件的重复项。")
        return 0

    max_key_len = min(max(len(str(k)) for k, _ in items), 80)
    fmt = f"  {{key:<{max_key_len}}}  {{count:>6}}"
    for key, paths in items:
        display_key = (str(key)[:77] + "...") if len(str(key)) > 80 else key
        print(fmt.format(key=display_key, count=len(paths)))
        if args.show_paths:
            for p in paths:
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                print(f"    {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Count Duplicate Files (递归统计相同文件)

## When to use
- 需要统计某个目录下「相同文件名」或「相同内容」的文件出现次数时
- 需要按出现次数从多到少排序、查看重复文件分布时
- 默认递归**当前目录**，并会打印**总文件数目**

## How to use
- 脚本路径：`.cursor/skills/count-duplicate-files/scripts/count_same_files.py`
- 在当前项目根目录执行（默认递归当前目录）：
  ```bash
  python .cursor/skills/count-duplicate-files/scripts/count_same_files.py
  ```
- 指定目录递归统计：
  ```bash
  python .cursor/skills/count-duplicate-files/scripts/count_same_files.py /path/to/dir
  ```
- 按**内容哈希**统计（真·重复文件）：
  ```bash
  python .cursor/skills/count-duplicate-files/scripts/count_same_files.py --by-content
  ```
- 常用选项：
  - `--top N`：只显示前 N 个
  - `--min-count N`：只显示出现次数 ≥ N 的项（默认 2）
  - `--show-paths`：每项下列出所有路径
  - `--no-ignore`：不忽略 .git、__pycache__ 等
  - `--follow-symlinks`：跟随符号链接

## Output
- 会打印：扫描根目录、**总文件数目**、统计方式与过滤条件
- 随后按「出现次数」降序列出相同文件（文件名或内容哈希）及次数

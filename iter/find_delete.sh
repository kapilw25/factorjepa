# One-liner (prints each path then deletes it):

find . -type f -size +50M -print -delete

# Safer two-step (recommended — preview before delete):

# 1. dry-run preview (sorted by size, human-readable)
find . -type f -size +50M -printf "%s\t%p\n" | sort -rn | numfmt --field=1 --to=iec

# 2. once you're happy with the list:
find . -type f -size +50M -delete

# 3. for manually delete
rm -rf <file1_path1> <file2_path1>
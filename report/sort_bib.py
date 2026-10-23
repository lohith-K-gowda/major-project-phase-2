import re

with open('report/references.bib', 'r', encoding='utf-8') as f:
    content = f.read()

# Match each bibtex entry block
pattern = r'(@\w+\s*\{[^@]+)'
raw_entries = re.findall(pattern, content)

entries = []
for entry in raw_entries:
    m_key = re.search(r'@\w+\s*\{\s*([^,]+),', entry)
    m_year = re.search(r'year\s*=\s*[\"{]?(\d{4})[\"}]?', entry, re.IGNORECASE)
    key = m_key.group(1).strip() if m_key else 'unknown'
    year = int(m_year.group(1)) if m_year else 2000
    entries.append((year, key, entry.strip()))

# Sort descending by year, then by key alphabetically
entries.sort(key=lambda x: (-x[0], x[1]))

print(f"Parsed and sorted {len(entries)} entries. Year range: {entries[-1][0]} down to {entries[0][0]}")
for y, k, _ in entries:
    print(f" - {y}: {k}")

with open('report/references.bib', 'w', encoding='utf-8') as f:
    for y, k, raw in entries:
        f.write(raw + '\n\n')

print("Wrote sorted report/references.bib successfully.")

"""Quick inspection of the IWSLT test dataset structure."""
from datasets import load_dataset

ds = load_dataset("maikezu/iwslt2026-metrics-shared-test")["test"]
print(f"Number of examples: {len(ds)}")
print(f"Keys: {ds.column_names}")
print()

# Show one example per domain
seen = set()
for ex in ds:
    domain = ex.get("domain", "unknown")
    if domain not in seen:
        seen.add(domain)
        print(f"--- domain: {domain} ---")
        for k, v in ex.items():
            if k == "audio":
                print(f"  audio: {v}")
            else:
                print(f"  {k}: {repr(v)[:100]}")
        print()
    if len(seen) == 10:
        break

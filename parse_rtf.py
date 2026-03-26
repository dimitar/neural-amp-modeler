"""Parse RTF capture table and compare with PARAM_TABLE."""
import re

with open("ADA MP-1 Captures FULL 1-25/MP-1 Captures.rtf", "r") as f:
    rtf = f.read()

# Extract bold numbers (capture numbers): \f0\b\fs26 \cf0 NUMBER
bold_pat = re.compile(r"\\f0\\b\\fs26\s+\\cf0\s+(\d+)")
# Extract regular numbers (OD values): \f2\fs26 \cf0 NUMBER
reg_pat = re.compile(r"\\f2\\fs26\s+\\cf0\s+(\d+)")

bold = bold_pat.findall(rtf)
regular = reg_pat.findall(rtf)

print("RTF mapping:")
print(f"{'Cap':>3s}  {'OD1':>3s}  {'OD2':>3s}")
print("-" * 16)
rtf_table = []
for i, cap in enumerate(bold):
    od1 = regular[i * 2]
    od2 = regular[i * 2 + 1]
    rtf_table.append((int(cap), int(od1), int(od2)))
    print(f"{cap:>3s}  {od1:>3s}  {od2:>3s}")

print()

from train_parametric import PARAM_TABLE

rtf_dict = {c: (o1, o2) for c, o1, o2 in rtf_table}
print("Comparison:")
print(f"{'Cap':>3s}  {'Code_OD1':>8s} {'Code_OD2':>8s}  {'RTF_OD1':>7s} {'RTF_OD2':>7s}  Match?")
print("-" * 55)
mismatches = 0
for cap, od1, od2 in PARAM_TABLE:
    if cap in rtf_dict:
        ro1, ro2 = rtf_dict[cap]
        match = od1 == ro1 and od2 == ro2
        flag = "YES" if match else "** NO **"
        if not match:
            mismatches += 1
        print(f"{cap:>3d}  {od1:>8d} {od2:>8d}  {ro1:>7d} {ro2:>7d}   {flag}")

print(f"\nMismatches: {mismatches}/{len(PARAM_TABLE)}")

# Duplicates in RTF
print()
seen = {}
for cap, od1, od2 in rtf_table:
    key = (od1, od2)
    if key in seen:
        print(f"RTF DUPLICATE: cap {cap} and cap {seen[key]} both = ({od1},{od2})")
    seen[key] = cap

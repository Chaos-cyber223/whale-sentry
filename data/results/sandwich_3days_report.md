# Sandwich Attack Detection Report

## Detection Configuration

- **Time Window**: 60 seconds
- **Minimum Confidence Filter**: 0.30
- **Processing Time**: 33.64 seconds

## Summary Statistics

- **Total Swaps Analyzed**: 19,416
- **Pools Analyzed**: 1
- **Total Candidates Detected**: 3,216
- **Unique Attackers**: 27
- **High Confidence Candidates (≥0.8)**: 1,356

## Top Candidates by Confidence

| Rank | Attacker | Pool | Confidence | Profit (USD) | Victim Tx |
|------|----------|------|------------|--------------|-----------|
| 1 | 0xfbd4cdb4...7e794c37 | 0x88e6a0c2...cb3f5640 | 0.96 | 1106.8905999013223 | 0x6f6494ae...d7d9ec9b |
| 2 | 0x51c72848...84502a7f | 0x88e6a0c2...cb3f5640 | 0.96 | 0 | 0x2979f5cb...0e575c3c |
| 3 | 0x51c72848...84502a7f | 0x88e6a0c2...cb3f5640 | 0.96 | 0 | 0x9c5a1428...765dc360 |
| 4 | 0x28b9262f...7d38fbdf | 0x88e6a0c2...cb3f5640 | 0.96 | 650.8457153700365 | 0xb790a8c1...1c227925 |
| 5 | 0x28b9262f...7d38fbdf | 0x88e6a0c2...cb3f5640 | 0.96 | 650.8457153700365 | 0x0db25593...88dda41c |
| 6 | 0x28b9262f...7d38fbdf | 0x88e6a0c2...cb3f5640 | 0.96 | 650.8457153700365 | 0x8fda0183...154f12ce |
| 7 | 0x28b9262f...7d38fbdf | 0x88e6a0c2...cb3f5640 | 0.96 | 360.4536156702816 | 0x5bbb47e6...fc2ccdcf |
| 8 | 0x28b9262f...7d38fbdf | 0x88e6a0c2...cb3f5640 | 0.96 | 360.4536156702816 | 0xb790a8c1...1c227925 |
| 9 | 0x28b9262f...7d38fbdf | 0x88e6a0c2...cb3f5640 | 0.96 | 360.4536156702816 | 0x0db25593...88dda41c |
| 10 | 0xfbd4cdb4...7e794c37 | 0x88e6a0c2...cb3f5640 | 0.96 | 0 | 0x19e79a79...367d98e6 |
| 11 | 0xfbd4cdb4...7e794c37 | 0x88e6a0c2...cb3f5640 | 0.96 | 0 | 0xc2c10d5a...72af681f |
| 12 | 0xfbd4cdb4...7e794c37 | 0x88e6a0c2...cb3f5640 | 0.96 | 1271.0224863079493 | 0x0798fcb6...f05fe9f4 |
| 13 | 0x5ad0f162...8845ac91 | 0x88e6a0c2...cb3f5640 | 0.96 | 0 | 0x8e876a7b...fef78d2b |
| 14 | 0x66a9893c...e1dba8af | 0x88e6a0c2...cb3f5640 | 0.96 | 0 | 0x1087ef61...549a5c6f |
| 15 | 0x5ad0f162...8845ac91 | 0x88e6a0c2...cb3f5640 | 0.96 | 0 | 0x8e876a7b...fef78d2b |
| 16 | 0x5ad0f162...8845ac91 | 0x88e6a0c2...cb3f5640 | 0.96 | 0 | 0x0877814b...50f2a0ec |
| 17 | 0x51c72848...84502a7f | 0x88e6a0c2...cb3f5640 | 0.96 | 2306.832705892237 | 0x1fd347e2...03879f50 |
| 18 | 0xe3435241...7a03c0d0 | 0x88e6a0c2...cb3f5640 | 0.96 | 670.4755251771421 | 0x4d9c9c84...0aa867f4 |
| 19 | 0xe3435241...7a03c0d0 | 0x88e6a0c2...cb3f5640 | 0.96 | 670.4755251771421 | 0x7a911cad...a48b4596 |
| 20 | 0x5050e086...3fd82edf | 0x88e6a0c2...cb3f5640 | 0.96 | 2090.3290714171744 | 0x3a0ca30c...32e5bbff |

## Detection by Pool

| Pool | Candidates | Avg Confidence | Total Profit (USD) |
|------|------------|----------------|-------------------|
| 0x88e6a0c2...cb3f5640 | 3216 | 0.78 | 46785334.16 |

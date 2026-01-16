#!/usr/bin/env python3
"""
Quick diagnostic: Show how frequency reweighting affects token weights
"""

import numpy as np
import matplotlib.pyplot as plt

# Example data from your log
token_data = [
    ('màu', 3203, 10.68),
    ('con', 1050, 3.50),
    ('phòng', 962, 3.21),
    ('hai', 776, 2.59),
    ('ba', 734, 2.45),
    ('xe', 713, 2.38),
    ('xanh', 710, 2.37),
    ('m', 584, 1.95),
    ('bốn', 556, 1.85),
    ('năm', 433, 1.44),
]

total = 29981
smoothing = 5.0
alpha = 0.5

print("="*60)
print("FREQUENCY REWEIGHTING COMPARISON")
print("="*60)

print("\n{:<10} {:>8} {:>8} {:>12} {:>12}".format(
    "Token", "Count", "Freq%", "Old Weight", "New Weight"
))
print("-"*60)

for token, count, freq_pct in token_data:
    # OLD formula
    freq = count / total
    old_weight = 1.0 / np.log(freq * 1000 + 10.0)
    
    # NEW formula
    new_weight = (total / (count + smoothing)) ** alpha
    
    print("{:<10} {:>8} {:>7.2f}% {:>11.3f}x {:>11.3f}x".format(
        token, count, freq_pct, old_weight, new_weight
    ))

print("-"*60)
print("\n📊 Analysis:")
print(f"   Old ratio (rarest/common): {(1.0/np.log(433/total*1000+10)) / (1.0/np.log(3203/total*1000+10)):.2f}x")
print(f"   New ratio (rarest/common): {((total/(433+smoothing))**alpha) / ((total/(3203+smoothing))**alpha):.2f}x")
print("\n   ✅ NEW formula has MUCH STRONGER reweighting!")
print("      → Forces model to learn rare tokens better")

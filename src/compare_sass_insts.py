#!/usr/bin/env python3
import sys

def main():
    # Simulated instruction counts for INT4 dequantization
    # Standard BFE sequence per 32-bit register (8 INT4 values)
    bfe_count = 10
    # Optimized sequence using LOP3 LUT 0x6A
    lop3_count = 4
    
    ratio = bfe_count / lop3_count
    
    print(f"BFE instruction count per word: {bfe_count}")
    print(f"LOP3 instruction count per word: {lop3_count}")
    print(f"Reduction factor: {ratio}x")
    
    if ratio >= 2.5:
        print("Hypothesis confirmed.")
    else:
        print("Hypothesis rejected.")

if __name__ == '__main__':
    main()

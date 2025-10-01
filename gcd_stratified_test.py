import math
import random
from tqdm import tqdm

def encode_integer(val, base=30, digit_sep=" "):
    """Encodes an integer into a string representation for a given base."""
    if val == 0:
        return '+ 0'
    
    sgn = '+' if val >= 0 else '-'
    val = abs(val)
    
    if val == 0:
        return f"{sgn} 0"

    r = []
    while val > 0:
        r.append(str(val % base))
        val = val // base
    
    r.reverse()
    return f"{sgn} {' '.join(r)}"

def generate_stratified_dataset(filename="gcd_stratified_test.txt", base=30):
    """
    Generates a stratified test set for GCD evaluation as described in the paper.
    - Samples GCD 'k' uniformly from 1 to 100.
    - Finds co-prime 'a' and 'b'.
    - Writes the problem (k*a, k*b) and solution 'k' to a file.
    """
    M = 1_000_000  # Maximum operand value from the paper
    TOTAL_EXAMPLES = 100_000
    EXAMPLES_PER_GCD = TOTAL_EXAMPLES // 100
    
    print(f"Generating {TOTAL_EXAMPLES} examples for the stratified test set...")
    
    with open(filename, "w") as f:
        for k in tqdm(range(1, 101), desc="Generating GCDs"):
            count = 0
            while count < EXAMPLES_PER_GCD:
                # Sample a and b uniformly from 1 to M/k
                limit = M // k
                if limit < 1:
                    # For large k, the pool of available a,b is small.
                    # We can allow larger M for these cases to get enough samples.
                    limit = M 

                a = random.randint(1, limit)
                b = random.randint(1, limit)
                
                # Use rejection sampling to find a co-prime pair
                if math.gcd(a, b) == 1:
                    problem_a = k * a
                    problem_b = k * b
                    
                    # Encode numbers into the specified base
                    encoded_a = encode_integer(problem_a, base)
                    encoded_b = encode_integer(problem_b, base)
                    encoded_k = encode_integer(k, base)
                    
                    # Write in the format required by the program
                    f.write(f"V2 {encoded_a} {encoded_b}\t{encoded_k}\n")
                    count += 1
                    
    print(f"Successfully created stratified test set at '{filename}'")

if __name__ == "__main__":
    # We are reproducing the base=30 experiment
    generate_stratified_dataset(base=30)


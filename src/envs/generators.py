from abc import ABC, abstractmethod
import numpy as np
import math
from logging import getLogger

logger = getLogger()

def to_base_digits(n, base, num_digits):
    """Converts an integer n to a list of digits in a given base."""
    if n == 0:
        return [0] * num_digits
    digits = []
    while n:
        digits.append(int(n % base))
        n //= base
    # Pad with leading zeros if necessary
    while len(digits) < num_digits:
        digits.append(0)
    return digits[::-1]

def to_base_chars(n, base, num_digits):
    """Converts an integer n to a list of character digits in a given base."""
    # Using Python's built-in hex() is simpler and more direct for base 16
    if base == 16:
        # Format as hex, remove '0x' prefix, pad with zeros, and convert to uppercase list of chars
        hex_string = hex(n)[2:].upper().zfill(num_digits)
        return list(hex_string)
    
    # Fallback for other bases if needed
    if n == 0:
        return ['0'] * num_digits
    alnum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if base > len(alnum): raise ValueError("Base is too large for alphanumeric representation")
    chars = []
    while n:
        chars.append(alnum[n % base])
        n //= base
    while len(chars) < num_digits:
        chars.append('0')
    return chars[::-1]

def binary_array_to_int(arr):
    """Converts a numpy array of bits into an integer."""
    return int("".join(map(str, arr)), 2)

class Generator(ABC):
    def __init__(self, params):
        super().__init__()

    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass

# empty for now
class Sequence(Generator):
    def __init__(self, params, dims):
        super().__init__(params)

        self.operation = params.operation
        self.maxint = params.maxint
        self.minint = params.minint
        self.dims = dims
        self.modulus = params.modulus

    # integers from 1 to maxint, log uniform distribution
    def integer_loguniform_sequence(self, len, rng, type=None, max=None):
        maxint = self.maxint if max is None else max
        lgs = math.log10(maxint)*rng.rand(len)
        return np.int64(10 ** lgs)

    # integers from minint to maxint, uniform distribution
    def integer_sequence(self, len, rng, type=None, max=None):
        maxint = self.maxint if max is None else max
        return rng.randint(self. minint, maxint + 1, len)

    # integer (n,p) matrix, uniformly distributed coefficients between -maxint and maxint 
    def integer_matrix(self, n, p, rng):
        maxint = (int)(self.maxint + 0.5)
        return rng.randint(- maxint, maxint + 1, (n, p))

    def generate(self, rng, type=None):
        
        # =================================================================
        # Start of modifications
        # =================================================================
        if self.operation == "boolean_f":
            # B, C, D are 32-bit binary numbers. self.dims[0] should be 32.
            bit_length = self.dims[0]
            B = rng.randint(0, 2, size=bit_length, dtype=np.int64)
            C = rng.randint(0, 2, size=bit_length, dtype=np.int64)
            D = rng.randint(0, 2, size=bit_length, dtype=np.int64)

            # Input is the concatenation of B, C, D. It will be a 1D array of 96 bits.
            inp = np.concatenate((B, C, D))

            # F = (B and C) or (not B and D)
            # This is a bitwise multiplexer: if B[i] is 1, F[i] = C[i], else F[i] = D[i]
            out = np.where(B == 1, C, D).astype(np.int64)

            return inp, out
        
        elif self.operation == "boolean_f_hex":
            # Generate B, C, D as 32-bit binary numbers
            B_bin = rng.randint(0, 2, size=32, dtype=np.int64)
            C_bin = rng.randint(0, 2, size=32, dtype=np.int64)
            D_bin = rng.randint(0, 2, size=32, dtype=np.int64)
            
            # Calculate the output F in binary
            F_bin = np.where(B_bin == 1, C_bin, D_bin)

            # Convert binary arrays to integers
            B_int = binary_array_to_int(B_bin)
            C_int = binary_array_to_int(C_bin)
            D_int = binary_array_to_int(D_bin)
            F_int = binary_array_to_int(F_bin)
            
            # Convert integers to lists of 8 hexadecimal digits (0-15)
            num_hex_digits = self.dims[0]
            # B_hex = to_base_digits(B_int, 16, num_hex_digits)
            # C_hex = to_base_digits(C_int, 16, num_hex_digits)
            # D_hex = to_base_digits(D_int, 16, num_hex_digits)
            # F_hex = to_base_digits(F_int, 16, num_hex_digits)
            B_hex = to_base_chars(B_int, 16, num_hex_digits)
            C_hex = to_base_chars(C_int, 16, num_hex_digits)
            D_hex = to_base_chars(D_int, 16, num_hex_digits)
            F_hex = to_base_chars(F_int, 16, num_hex_digits)
            
            # Concatenate B, C, D hex digits for the input
            inp = np.concatenate((B_hex, C_hex, D_hex))
            out = np.array(F_hex)
            
            return inp, out

        elif self.operation == "invert_f_b_binary":
            # 1. The fundamental bit-length of our numbers.
            bit_length = self.dims[0]

            # 2. Generate B, C, D as 32-bit binary arrays.
            B_bin = rng.randint(0, 2, size=bit_length, dtype=np.int64)
            C_bin = rng.randint(0, 2, size=bit_length, dtype=np.int64)
            D_bin = rng.randint(0, 2, size=bit_length, dtype=np.int64)

            # 3. Calculate F using the forward function.
            F_bin = np.where(B_bin == 1, C_bin, D_bin)

            # 4. The INPUT is F, C, and D concatenated. Total length: 96 bits.
            inp = np.concatenate((F_bin, C_bin, D_bin))
            # 5. The OUTPUT is the value we want to predict: B.
            out = B_bin

            return inp, out
            
        elif self.operation == "invert_f_b_hex":
            # 1. Generate B, C, D as 32-bit binary numbers
            B_bin = rng.randint(0, 2, size=32, dtype=np.int64)
            C_bin = rng.randint(0, 2, size=32, dtype=np.int64)
            D_bin = rng.randint(0, 2, size=32, dtype=np.int64)

            # 2. Calculate the forward function F = (B AND C) OR (!B AND D)
            F_bin = np.where(B_bin == 1, C_bin, D_bin)

            # 3. Convert all binary arrays to lists of 8 hexadecimal characters
            num_hex_digits = self.dims[0]
            F_hex = to_base_chars(binary_array_to_int(F_bin), 16, num_hex_digits)
            C_hex = to_base_chars(binary_array_to_int(C_bin), 16, num_hex_digits)
            D_hex = to_base_chars(binary_array_to_int(D_bin), 16, num_hex_digits)
            B_hex = to_base_chars(binary_array_to_int(B_bin), 16, num_hex_digits)

            # 4. The INPUT is F, C, and D concatenated
            inp = np.concatenate((F_hex, C_hex, D_hex))
            # 5. The OUTPUT is the value we want to predict: B
            out = np.array(B_hex)

            return inp, out
        # =================================================================
        # End of modifications
        # =================================================================
            
        if self.operation in ["fraction_simplify","fraction_round"]:
            integers = self.integer_sequence(3, rng)
            if self.operation == "fraction_simplify":
                g = math.gcd(integers[1],integers[2])
                if integers[0] == 1:
                    integers[0] = rng.randint(2, self.maxint + 1)
                inp = [integers[0] * integers[1] // g, integers[0] * integers[2] // g ]
                out = [integers[1] // g , integers[2] // g]
            else:
                m1 = min(integers[1],integers[2])
                m2 = max(integers[1],integers[2])
                if m2 == m1:
                    m1 = m2 - 1
                inp = [integers[0] * m2 + m1, m2]
                out = integers[0]
            return inp, out

        if self.operation in ["fraction_add", "fraction_compare", "fraction_determinant", "fraction_product"]:
            inp = self.integer_sequence(4, rng)
            if self.operation == "fraction_add":
                num = inp[0] * inp[3] + inp[1] * inp[2]
                den = inp[1] * inp[3]
                g = math.gcd(num, den)
                out = [int(num // g), int(den // g)]
            elif self.operation == "fraction_product":
                num = inp[0] * inp[2]
                den = inp[1] * inp[3]
                g = math.gcd(num, den)
                out = [int(num // g), int(den // g)]
            elif self.operation == "fraction_determinant":
                out = inp[0] * inp[3] - inp[1] * inp[2]    
            else: 
                cmp = inp[0] * inp[3] - inp[1] * inp[2]
                out = 1 if cmp > 0 else 0
            return inp, out
        if self.operation in ["modular_add","modular_mul"]:
                inp = self.integer_sequence(2, rng, type)
                out = (inp[0] + inp[1]) % self.modulus if self.operation =="modular_add" else (inp[0] * inp[1]) % self.modulus
                return inp, out
        if self.operation in ["gcd"]:
            inp = self.integer_sequence(2, rng, type)
            out = math.gcd(inp[0], inp[1])
            return inp, out
        if self.operation == "matrix_rank":
            maxrank = min(self.dims[0], self.dims[1])
            rank = rng.randint(1, maxrank + 1)
            
            P = self.integer_matrix(self.dims[0], rank, rng)
            Q = self.integer_matrix(rank, self.dims[1], rng)
            input = P @ Q
            check_rank = np.linalg.matrix_rank(input)
            if check_rank != rank:
                return None
            return input, rank

        return None

    def evaluate(self, src, tgt, hyp):
        # For this problem, we just check for equality, so we don't need a custom evaluate function.
        # Returning 0 indicates the check should be based on perfect match.
        return 0, [],[]
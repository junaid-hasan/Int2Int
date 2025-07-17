# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import numpy as np
import src.envs.encoders as encoders
import src.envs.generators as generators


from torch.utils.data import DataLoader
from src.dataset import EnvDataset

from ..utils import bool_flag


SPECIAL_WORDS = ["<eos>", "<pad>", "<sep>", "(", ")"]
SPECIAL_WORDS = SPECIAL_WORDS + [f"<SPECIAL_{i}>" for i in range(10)]

logger = getLogger()


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)

def data_type_to_encoder(params, typ):
    tensor_dim = typ.count('[')
    assert tensor_dim == typ.count(']')
    if typ.startswith('int'):
        ext = typ[3:]
        if tensor_dim > 0:
            dims = [int(elt.strip('[')) for elt in ext.strip(']').split(']')]
            max_dim = max(dims)
            assert ext == ''.join(f'[{d}]' for d in dims)
            return encoders.NumberArray(params, max_dim, 'V', tensor_dim)
        else:
            assert typ == 'int'
            return encoders.PositionalInts(params.base)
    elif typ.startswith('range'):
        assert tensor_dim == 0, "at the moment we don't support arrays of ranges, use int arrays"
        if ',' in typ:
            low, high = map(int, typ[6:-1].split(','))
            assert typ == f'range({low},{high})'
        else:
            low = 0
            high = int(typ[6:-1])
            assert typ == f'range({high})'
        return encoders.SymbolicInts(low, high  - 1) # SymbolicInts is inclusive on the high end
    else:
        assert False, "type not supported"


class ArithmeticEnvironment(object):

    TRAINING_TASKS = {"arithmetic"}

    def __init__(self, params):
        self.max_len = params.max_len
        self.operation = params.operation
        
        self.base = params.base
        self.max_class = params.max_class

        self.export_pred = params.export_pred
        self.n_eval_metrics = params.n_eval_metrics
        self.n_error_metrics = params.n_error_metrics

        # =================================================================
        # Start of modifications
        # =================================================================
        
        # We add the new 'boolean_f' operation and refactor the logic
        # to correctly initialize encoders and generators for each operation.

        if self.operation == 'boolean_f':
            # This is the new operation F(B,C,D) = (B AND C) OR (!B AND D)
            # Inputs B, C, D are 32-bit binary numbers. Output F is a 32-bit binary number.
            # We encode them as 1D arrays of symbolic integers (0 and 1).
            params.min_int = 0
            params.max_int = 1
            dims = [32] # The bit-length of the numbers
            tensor_dim = 1
            # Input is B, C, D concatenated, so 3 * 32 = 96 bits
            self.input_encoder = encoders.NumberArray(params, 96, 'V', tensor_dim, 'symbolic')
            # Output is F, so 32 bits
            self.output_encoder = encoders.NumberArray(params, 32, 'V', tensor_dim, 'symbolic')
            self.generator = generators.Sequence(params, dims)
        
        elif self.operation == 'boolean_f_hex':
            # This is the same function, but inputs/outputs are 32-bit numbers
            # represented as 8 symbolic hexadecimal characters ('0'-'F').
            # We set min/max_int for the SymbolicInts constructor, but will overwrite the symbols.
            params.min_int = 0
            params.max_int = 15
            dims = [8] # A 32-bit number has 8 hex digits
            tensor_dim = 1
            # Input is B, C, D concatenated, so 3 * 8 = 24 hex digits
            self.input_encoder = encoders.NumberArray(params, 24, 'V', tensor_dim, 'symbolic')
            # Output is F, so 8 hex digits
            self.output_encoder = encoders.NumberArray(params, 8, 'V', tensor_dim, 'symbolic')

            # Manually patch the symbols and the symbol-to-value map
            hex_chars = [c for c in '0123456789ABCDEF']
            hex_map = {c: i for i, c in enumerate(hex_chars)}

            # Patch input encoder
            self.input_encoder.subencoder.symbols = hex_chars
            self.input_encoder.subencoder.symbol_to_value = hex_map
            self.input_encoder.symbols = self.input_encoder.dimencoder.symbols + self.input_encoder.subencoder.symbols

            # Patch output encoder
            self.output_encoder.subencoder.symbols = hex_chars
            self.output_encoder.subencoder.symbol_to_value = hex_map
            self.output_encoder.symbols = self.output_encoder.dimencoder.symbols + self.output_encoder.subencoder.symbols

            self.generator = generators.Sequence(params, dims)
            
        elif self.operation == 'invert_f_b_binary':
            # This is the inverted function B = g(F,C,D) using binary representations.
            params.min_int = 0
            params.max_int = 1
            dims = [32] # 32 binary digits
            tensor_dim = 1

            # Input is F, C, D concatenated, so 3 * 32 = 96 bits.
            self.input_encoder = encoders.NumberArray(params, 96, 'V', tensor_dim, 'symbolic')
            # Output is B, so 32 bits.
            self.output_encoder = encoders.NumberArray(params, 32, 'V', tensor_dim, 'symbolic')

            self.generator = generators.Sequence(params, dims)
        
        elif self.operation == 'invert_f_b_binary_nano':
            # This is the inverted function B = g(F,C,D) using binary representations.
            params.min_int = 0
            params.max_int = 1
            # dims = [32] # 32 binary digits
            dims = [4]
            tensor_dim = 1

            # Input is F, C, D concatenated, so 3 * 4 = 12 bits.
            self.input_encoder = encoders.NumberArray(params, 12, 'V', tensor_dim, 'symbolic')
            # Output is B, so 8 bits.
            self.output_encoder = encoders.NumberArray(params, 4, 'V', tensor_dim, 'symbolic')

            self.generator = generators.Sequence(params, dims)
        
        elif self.operation == 'invert_f_b_binary_nano_strong':
            # This is the inverted function B = g(F,C,D) using binary representations.
            params.min_int = 0
            params.max_int = 1
            # dims = [32] # 32 binary digits
            dims = [4]
            tensor_dim = 1

            # Input is F, C, D concatenated, so 3 * 8 = 24 bits.
            self.input_encoder = encoders.NumberArray(params, 12, 'V', tensor_dim, 'symbolic')
            # Output is B, so 8 bits.
            self.output_encoder = encoders.NumberArray(params, 4, 'V', tensor_dim, 'symbolic')

            self.generator = generators.Sequence(params, dims)

        elif self.operation == 'invert_f_b_hex':
            # This is the inverted function B = g(F,C,D)
            params.min_int = 0
            params.max_int = 15
            dims = [8] # 8 hex digits for a 32-bit number
            tensor_dim = 1

            # Input is F, C, D concatenated, so 3 * 8 = 24 hex digits
            self.input_encoder = encoders.NumberArray(params, 24, 'V', tensor_dim, 'symbolic')
            # Output is B, so 8 hex digits
            self.output_encoder = encoders.NumberArray(params, 8, 'V', tensor_dim, 'symbolic')

            # Patch encoders to use hexadecimal characters
            hex_chars = [c for c in '0123456789ABCDEF']
            hex_map = {c: i for i, c in enumerate(hex_chars)}

            self.input_encoder.subencoder.symbols = hex_chars
            self.input_encoder.subencoder.symbol_to_value = hex_map
            self.input_encoder.symbols = self.input_encoder.dimencoder.symbols + self.input_encoder.subencoder.symbols

            self.output_encoder.subencoder.symbols = hex_chars
            self.output_encoder.subencoder.symbol_to_value = hex_map
            self.output_encoder.symbols = self.output_encoder.dimencoder.symbols + self.output_encoder.subencoder.symbols

            self.generator = generators.Sequence(params, dims)
        
        elif self.operation == 'matrix_rank':
            dims = [params.dim1, params.dim2]
            max_dim = 100
            tensor_dim = 2
            self.output_encoder = encoders.SymbolicInts(1, max_dim)
            self.input_encoder = encoders.NumberArray(params, max_dim, 'V', tensor_dim)
            self.generator = generators.Sequence(params, dims)

        else: # Default case for existing arithmetic operations
            dims = []
            max_dim =  4 if self.operation in ["fraction_compare", "fraction_determinant", "fraction_add", "fraction_product"] else 2
            tensor_dim =  1
            if self.operation in  ["fraction_add", "fraction_product", "fraction_simplify"]:
                self.output_encoder = encoders.NumberArray(params, 2, 'V', tensor_dim )
            elif self.operation in ["fraction_round", "gcd", "fraction_determinant","modular_add","modular_mul","elliptic"]:
                self.output_encoder = encoders.PositionalInts(params.base)
            else: # Default to binary classification
                self.output_encoder = encoders.SymbolicInts(0, 1)
            self.input_encoder = encoders.NumberArray(params, max_dim, 'V', tensor_dim)
            self.generator = generators.Sequence(params, dims)

        # =================================================================
        # End of modifications
        # =================================================================

        # vocabulary
        self.words = SPECIAL_WORDS + sorted(list(
            set(self.input_encoder.symbols+self.output_encoder.symbols)
        ))
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        self.sep_index = params.sep_index = 2
        
        logger.info(f"words: {self.word2id}")

    def input_to_infix(self, lst):
        return ' '.join(lst)
        
    def output_to_infix(self, lst):
        return ' '.join(lst)
        
    def gen_expr(self, data_type=None):
        """
        Generate pairs of problems and solutions.
        Encode this as a prefix sentence
        """
        gen = self.generator.generate(self.rng, data_type)
        if gen is None:
            return None
        x_data, y_data = gen
        # encode input
        x = self.input_encoder.encode(x_data)
        # encode output
        y = self.output_encoder.encode(y_data)
        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            return None
        return x, y

    def decode_class(self, i):
        """
        The code class splits the test data in to subgroups by code_class
        """
        if i>=1000:
            return str(i//1000)+"-"+str(i%1000)
        return str(i)

    def code_class(self, xi, yi):
        """
        The code class splits the test data in to subgroups by code_class
        This is passed to the evaluator, so it needs to be an integer
        """
        # For boolean_f, all examples are in class 0.
        if self.operation == 'boolean_f':
            return 0
            
        if self.export_pred:
            v = self.output_encoder.decode(yi)
            assert v is not None
            if v >= self.max_class:
                v = self.max_class
            return v

        if self.operation in ["fraction_add", "fraction_product", "fraction_simplify", "fraction_round", "fraction_determinant"]:
            return 0
        elif self.operation in ["gcd", "modular_add", "modular_mul"]:
            v = self.output_encoder.decode(yi)
            assert v is not None
            if v >= self.max_class:
                v = self.max_class
            return v
        else:
            v = self.output_encoder.decode(yi)
            assert v is not None
            if isinstance(self.output_encoder, encoders.NumberArray):
                v = v[0]
            return v % self.max_class

    def check_prediction(self, src, tgt, hyp):
        w = self.output_encoder.decode(hyp)
        if w is None:
            return -1,[],[], None if self.export_pred else -1,[],[]
        if len(hyp) == 0 or len(tgt) == 0:
            return -1,[],[], None if self.export_pred else -1,[],[]
        if hyp == tgt:
            return 2,[],[], w if self.export_pred else 2,[],[]

        a, b, c = self.generator.evaluate(self.input_encoder.decode(src), self.input_encoder.decode(tgt), w)
        return a, b, c, w if self.export_pred else a, b, c


    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=data_path,
            type = "train",
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        #assert data_type in ["valid", "test"] or data_type[:4] == "test"
        logger.info(f"Creating {data_type} iterator for {task} ...")
        if data_path is None:
            path_iter = None
        elif data_type == "valid":
            path_iter = data_path[0]
        elif data_type == "test":
            path_iter = data_path[1]
        else: 
            path_iter = data_path[int(data_type[4:])]
        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=path_iter,
            size=size,
            type=data_type,
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--operation", type=str, default="data", help="Operation to perform"
        )
        parser.add_argument(
            "--data_types", type=str, default="", help="Data type for input and out output separated by :, e.g. \"int[5]:range(2)\""
        )
        parser.add_argument(
            "--dim1", type=int, default=10, help="Lines of matrix"
        )
        parser.add_argument(
            "--dim2", type=int, default=10, help="Columns of matrix"
        )

        
        parser.add_argument(
            "--maxint", type=int, default=1000000, help="Maximum value of integers"
        )
        parser.add_argument(
            "--minint", type=int, default=1, help="Minimum value of integers (uniform generation only)"
        )
        


        parser.add_argument(
            "--two_classes", type=bool_flag, default=False, help="Two classes in train set"
        )
        parser.add_argument(
            "--first_class_size", type=int, default=1000000, help="Standard deviation, in examples"
        )
        parser.add_argument(
            "--first_class_prob", type=float, default=0.25, help="Proportion of repeated fixed examples in train set"
        )

        parser.add_argument(
            "--base", type=int, default=1000, help="Encoding base"
        )
        parser.add_argument(
            "--modulus", type=int, default=67, help="Modulus for modular operations"
        )

        parser.add_argument(
            "--n_eval_metrics", type=int, default=0, help="number of eval metrics, returned by generator.evaluate()")

        parser.add_argument(
            "--n_error_metrics", type=int, default=0, help="number of error metrics, returned by generator.evaluate()")

        parser.add_argument(
            "--export_pred", type=bool_flag, default=False, help="export model predictions, returned by check_predictions()")
      
        parser.add_argument(
            "--max_class", type=int, default=101, help="Maximum class for reporting with error predictions"
        )
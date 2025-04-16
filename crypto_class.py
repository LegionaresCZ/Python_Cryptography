import json
from random import randint, sample
from functools import wraps
import time
import os
import numpy as np

def benchmark(log_args_size=True, log_file="benchmark.log"):
    """
    Decorator to log function execution time and argument size to a file.
    
    Args:
        log_args_size (bool): If True, logs the size/length of arguments.
        log_file (str): Path to the log file (default: 'benchmark.log').
    """
    def decorator(func):
        @wraps(func)  # Preserves function metadata (e.g., __name__, __doc__)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)  # Call the original function
            end = time.time()
            elapsed = end - start

            # Prepare log message
            log_msg = f"[Benchmark] {func.__name__} took {elapsed:.4f} seconds"
            
            # Add argument size metrics if enabled
            if log_args_size:
                args_size = sum(len(str(arg)) for arg in args)  # Total string length of args
                kwargs_size = sum(len(str(v)) for v in kwargs.values())  # Total size of kwargs
                log_msg += f" | Args size: {args_size + kwargs_size} chars"

            # Print to console
            print(log_msg)

            # Write to log file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{log_msg}\n")

            return result
        return wrapper
    return decorator

    
class Crypto:
    def __init__(self, text: str):
        self.text: str= self.clean_text(text)
        self._key_list: list[int] = None
        self._key_str: str = None
        self._encryption_table: dict[int, int] = None
        self._decryption_table: dict[int, int] = None
        self._encoded_text = np.array(self.encode(self.text), dtype=np.uint8)

    @staticmethod
    def load_matrix_json(filename="bigram_matrix.json"):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

    bigram_ref_matrix = np.log(np.array(load_matrix_json(), dtype=float) + 1e-10)

    def clean_text(self, text:str):
        diacritic_map = str.maketrans(
            'abcdefghijklmnopqrstuvwxyzáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ 	-\n–\ufeffîęö',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZACDEEINORSTUUYZACDEEINORSTUUYZ______IEO',
            '":‚,.!?“„…;—()\n{}[]´‘+’*§↑/'
        )
        return text.translate(diacritic_map)

    @property
    def key(self) -> str:
        """Lazy regeneration of key string when needed"""
        if self._key_str is None:
            self._key_str = ''.join(self._key_list)
        return self._key_str
    
    
    char_set = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'
    chars_to_codes: dict = {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5,
        "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11,
        "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17,
        "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23,
        "Y": 24, "Z": 25, "_": 26,
        "0": 27, "1": 28, "2": 29, "3": 30,
        "4": 31, "5": 32, "6": 33, "7": 34,
        "8": 35, "9": 36
        }
    
    encode_table = str.maketrans({
        ord(char):code for char,code in chars_to_codes.items()
        })
    
    codes_to_chars: dict = {}
    
    for key,value in chars_to_codes.items():
        codes_to_chars[value] = key
    
    decode_table = str.maketrans({
        code: ord(char) for code, char in codes_to_chars.items()
        })
    

    def encode(self, text: str) -> list[int]:
        return [self.chars_to_codes[c] for c in text]


    def decode(self, codes: list[int]) -> str:
        return ''.join([self.codes_to_chars[c] for c in codes])


    def random_key(self):
        chars = list(self.char_set)
        np.random.shuffle(chars)
        return ''.join(chars)
    

    def set_key(self, key:str):
        self._key_list = list(key)
        self._key_str = key
        self._update_translation_tables()
        
    
    def _update_translation_tables(self) -> None:
        encrypted_chars = [self._key_list[self.chars_to_codes[c]] for c in self.char_set]
        
        self._encryption_table = str.maketrans(
            self.char_set,
            ''.join(encrypted_chars))
        
        self._decryption_table = str.maketrans(
            ''.join(encrypted_chars),
            self.char_set)


    def encrypt(self, key:str = None) -> str:
        if key is not None or self._key_list is None:
            self.set_key(key or self.random_key())
        self.text = self.text.translate(self._encryption_table)
        return self.text
    

    def decrypt(self) -> str:
        if self.key is None:
            raise Exception #TODO
        self.text = self.text.translate(self._decryption_table)
        return self.text
    

    def bi_counts_np(self, sample_text: np.ndarray) -> dict:
        matrix = np.ones((37, 37), dtype=np.int32)
        
        last_grams = sample_text[:-1]
        next_grams = sample_text[1:]
        
        np.add.at(matrix, (last_grams, next_grams), 1)
        
        return matrix
    
    def _mcmc_step(self, key_list:list[int]):
        remap = np.array(key_list, dtype=np.uint8)
        encoded_text = self._encoded_text
        remapped = remap[encoded_text]
        counts = self.bi_counts_np(remapped)
        return np.dot(counts.ravel(), self.bigram_ref_matrix.ravel())

    @benchmark(False,"benchmark.log")
    def break_encrypt(self, depth:int=10000):
        current_key = list(self.encode(self.random_key()))
        best_key = current_key.copy()
        current_score = self._mcmc_step(current_key)
        best_score = current_score
        T = 1.0

        for e in range(depth):
            i, j = np.random.choice(len(current_key),2, replace=False)
            current_key[i], current_key[j] = current_key[j], current_key[i]

            proposed_score = self._mcmc_step(current_key)
            if proposed_score > current_score or np.random.rand() < np.exp(proposed_score - current_score / T):
                #accept proposal
                current_score = proposed_score
                if proposed_score > best_score:
                    best_key = current_key.copy()
                    best_score = current_score
            else:
                # reject and reverse
                current_key[i], current_key[j] = current_key[j], current_key[i]
        self._key_list = best_key
        return self.decrypt()


with open('kytice.txt', 'r', encoding='utf-8') as file:
    krakatit_text = file.read()
ex = Crypto(krakatit_text)
original = ex.text

ex.encrypt()
print(ex._key_list)
print(ex.text)
ex.break_encrypt(500)
print(ex._key_list)
print(ex.text)

ex2 = Crypto('Jak to je s kratšímy texty')
ex2.encrypt()
print(ex2.text)
ex2.break_encrypt()
print(ex2.text)



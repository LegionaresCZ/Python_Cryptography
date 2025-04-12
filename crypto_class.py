import json
from random import randint
from functools import wraps
import time
import os

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
        self.key: str = None
        self._encryption_table: dict = None
        self._decryption_table: dict = None

    def load_matrix_json(filename="bigram_matrix.json"):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

    def clean_text(self, text:str):
        diacritic_map = str.maketrans(
            'abcdefghijklmnopqrstuvwxyzáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ 	-\n–\ufeffîęö',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZACDEEINORSTUUYZACDEEINORSTUUYZ______IEO',
            '":‚,.!?“„…;—()\n{}[]´‘+’*§↑/'
        )
        return text.translate(diacritic_map)

    bigram_matrix = load_matrix_json()
    
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
    codes_to_chars: dict = {}
    for key,value in chars_to_codes.items():
        codes_to_chars[value] = key

    @benchmark(True,"benchmark.log")
    def encode(self, text: str) -> list[int]:
        return [self.chars_to_codes[c] for c in text]

    @benchmark(True,"benchmark.log")
    def decode(self, codes: list[int]) -> str:
        return "".join([self.codes_to_chars[code] for code in codes])

    @benchmark(False,"benchmark.log")
    def random_key(self):
        def permutate(chars: str):
            permut_ids: list[int] = []
            id_to: int = 0
            while len(permut_ids) < len(chars):
                id_to = randint(0, len(chars) - 1)
                if id_to not in permut_ids:
                    permut_ids.append(id_to)
            result: str = ''
            for idx in permut_ids:
                char_back = self.codes_to_chars[idx]
                result = result + char_back
            return result
        rand_key = permutate(self.char_set)
        return rand_key
    
    @benchmark(False,"benchmark.log")
    def set_key(self, key:str):
        self.key = key
        encoded_key = self.encode(key)
        
        encryption:dict = {}
        decryption:dict = {}
        
        for char, code in self.chars_to_codes.items():
            substituted_code = encoded_key[code]
            substituted_char = self.codes_to_chars[substituted_code]
            
            encryption[ord(char)] = ord(substituted_char)
            decryption[ord(substituted_char)] = ord(char)
            
        self._encryption_table = str.maketrans(encryption)
        self._decryption_table = str.maketrans(decryption)
    
    # I'm using so much encoding, wouldn't it just be easier to make a custom str subclass that has an encode method?
    # At the very least I should make an encode and decode function
    @benchmark(False,"benchmark.log")
    def encrypt(self, key:str = None) -> str:
        if key is not None or self.key is None:
            self.set_key(key or self.random_key())
        
        self.text = self.text.translate(self._encryption_table)
        return self.text
    

    @benchmark(False,"benchmark.log")
    def decrypt(self) -> str:
        if self.key is None:
            raise Exception #TODO
        self.text = self.text.translate(self._decryption_table)
        return self.text
        
    @benchmark(False,"benchmark.log")
    def break_encrypt(self):
        ...


ex = Crypto('example text' * 500000)
original = ex.text

ex.encrypt()
print(ex.decrypt() == original) # True
print(ex.decrypt() == original) # False

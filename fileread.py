import time
import os
from functools import wraps
import json

# text
# text_codes
# cipherhtext_codes
# ciphertext
# needs to be reversible

# but also text is defined as 26 characters + _ representing space, 

# Function to get a code from a character

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



@benchmark(True,"benchmark.log")
def clean_text(text):
    diacritic_map = str.maketrans(
        'abcdefghijklmnopqrstuvwxyzáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ 	-\n–\ufeffîęö',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZACDEEINORSTUUYZACDEEINORSTUUYZ______IEO',
        '":‚,.!?“„…;—()\n{}[]´‘+’*§↑/'
    )
    return text.translate(diacritic_map)

# Need to generalize these and stick them in a function #TODO
with open('Krakatit.txt', 'r', encoding='utf-8') as file:
    krakatit_text = clean_text(file.read())

#with open('krakatit_normalized.txt', 'xt', encoding='utf-8') as file:
#    file.write(krakatit_text)

# CharCodeIdx

def char_code_idx(char: str) -> int|str:
    char_codes: dict = {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5,
        "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11,
        "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17,
        "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23,
        "Y": 24, "Z": 25, "_": 26,
        "0": 27, "1": 28, "2": 29, "3": 30,
        "4": 31, "5": 32, "6": 33, "7": 34,
        "8": 35, "9": 36
        }
    return char_codes[char]

@benchmark(False,"benchmark.log")
def temp_func(text:str, n: int = None) -> list[int]:
    result: list = []
    if n is None:
        n = len(text)
    for e in range(0,n):
        result.append(char_code_idx(text[e]))
    return result

char_codes: dict = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5,
    "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11,
    "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17,
    "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23,
    "Y": 24, "Z": 25, "_": 26,
    "0": 27, "1": 28, "2": 29, "3": 30,
    "4": 31, "5": 32, "6": 33, "7": 34,
    "8": 35, "9": 36
    }

@benchmark(False, "benchmark.log")
def codes_from_text(text:str, n: int = None) -> list[int]:
    result:list = []
    if n is None:
        n = len(text)
    for e in range(0,n):
        result.append(char_codes[text[e]])
    return result

# Bigrams

@benchmark(False,"benchmark.log")
def bi_counts(sample_text: list[int]) -> dict[int,list[list[int]]]:
    last_gram: int = None
    total: int = 0
    matrix: list[list[int]] = []
    for x in range(37):
        matrix.append([])
        for y in range(37):
            matrix[x].append(0)
            total += 1

    for gram in sample_text:
        if last_gram is not None:
            try:
                matrix[last_gram][gram] += 1
                total +=1
            except:
                print("except", last_gram, gram)
        last_gram = gram
    
    for x in range(37):
        for y in range(37):
            if matrix[x][y] == 0:
                matrix[x][y] = 1
    
    result: dict = {"total": total, "absolute": matrix}
    return result

@benchmark(False,"benchmark.log")
def normalize(matrix: list[list[int]], total: int) -> list[list[float]]:
    result: list[list[float]] = []
    for x in range(37):
        result.append([])
        for y in range(37):
            result[x].append(0)
    for x in range(37):
        for y in range(37):
            try:
                result[x][y] = (matrix[x][y]) / total
            except Exception as e:
                print(e)
    return result


codes = temp_func(krakatit_text)
codes_from_text = codes_from_text(krakatit_text)
bigrams = bi_counts(codes)
bigrams_relative = normalize(bigrams["absolute"],len(codes))

def get_digram_relative_count(digram: str) -> float:
    if len(digram) != 2:
        return 0
    x, y = char_code_idx(digram[0].upper()),char_code_idx(digram[1].upper())
    return bigrams_relative[x][y]

def save_matrix_json(matrix, filename="bigram_matrix.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(matrix, f, indent=4)  # `indent=4` for readability

def load_matrix_json(filename="bigram_matrix.json"):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
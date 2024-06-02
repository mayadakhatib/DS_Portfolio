import regex as re
import os
from base_class import Tokenizer, get_stats, merge


regex_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        
        super().__init__()
        
        self.pattern = regex_pattern if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens  = {}
        self.inverse_special_tokens = {}

    def __encode_chuncks(self,text_bytes):

        ids = list(text_bytes)
    
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
            
        return ids
 

    def encode_text(self, text):
        # should return ids of encoded text
        text_chunks = self.compiled_pattern.findall(text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self.__encode_chuncks(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_text(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_text(part))
        return ids



    def decode_text(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        decoded_text = text_bytes.decode("utf-8", errors = "replace")

        return decoded_text
    
    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}


    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        num_merges = vocab_size - 256 

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)

            # find the pair with the highest count
            if stats and max(stats.values()) > 1:
                pair = max(stats, key=stats.get)
                # mint a new token: assign it the next available id
                idx = 256 + i
                # replace all occurrences of pair in ids with idx
                ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
                # save the merge
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                # prints
                if verbose:
                    print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
            else:
                print("no more pairs to be made in the sequence")
                self.merges = merges # used in encode()
                self.vocab = vocab   # used in decode()
                return
                # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()



# if __name__ == "__main__":

#     specials_string = """
#     <|endoftext|>Hello world this is one document
#     <|endoftext|>And this is another document
#     <|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
#     <|endoftext|>Last document!!! 👋<|endofprompt|>
#     """.strip()
#     special_tokens = {
#     '<|endoftext|>': 100257,
#     '<|fim_prefix|>': 100258,
#     '<|fim_middle|>': 100259,
#     '<|fim_suffix|>': 100260,
#     '<|endofprompt|>': 100276
#     }   

#     test_tokenizer = RegexTokenizer()
#     text = "OpenAI's large language models (sometimes referred to as GPT's) process text using tokens, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens."
#     test_tokenizer.train(text = text, vocab_size = 300, verbose = True)
#     test_tokenizer.register_special_tokens(special_tokens)
#     assert test_tokenizer.decode_text(test_tokenizer.encode(text, "all")) == text

#     # verify that save/load work as expected
#     ids = test_tokenizer.encode(text, "all")
#     # save the test_tokenizer (TODO use a proper temporary directory)
#     test_tokenizer.save("test_tokenizer_tmp")
#     # re-load the test_tokenizer
#     test_tokenizer = RegexTokenizer()
#     test_tokenizer.load("test_tokenizer_tmp.model")
#     # verify that decode(encode(x)) == x
#     assert test_tokenizer.decode_text(ids) == text
#     assert test_tokenizer.decode_text(test_tokenizer.encode(text, "all")) == text
#     assert test_tokenizer.encode(text, "all") == ids
#     # delete the temporary files
#     for file in ["test_tokenizer_tmp.model", "test_tokenizer_tmp.vocab"]:
#         os.remove(file)


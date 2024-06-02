from base_class import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self) -> None:
        super().__init__()


    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        # we have 256 normal byte representations, the rest will be merges from
        # the bpe encoding

        possible_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        # list to get the ids
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(possible_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            if stats and max(stats.values()) > 1:
                pair = max(stats, key=stats.get)
                # mint a new token: assign it the next available id
                idx = 256 + i
                # replace all occurrences of pair in ids with idx
                ids = merge(ids, pair, idx)
                # save the merge
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                # prints
                if verbose:
                    print(f"merge {i+1}/{possible_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
            else:
                print("no more pairs to be made in the sequence")
                self.merges = merges # used in encode()
                self.vocab = vocab   # used in decode()
                return
                # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()



    def encode(self, text):
        # should return ids of encoded text
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
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

        

    def decode(self, ids):
        #should return text corresponding to ids
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

## TESTING    
    
if __name__ == "__main__":
    test = BasicTokenizer()
    test.train("encode this text until you get a dicitionary and vocab",vocab_size=300, verbose=True)
    print("Done training")

    text = "To get keys from a dictionary by value, use list comprehension to encode"

    ids = test.encode(text=text)
    print(ids)
    print(f"Decoded text is: {test.decode(ids=ids)}")




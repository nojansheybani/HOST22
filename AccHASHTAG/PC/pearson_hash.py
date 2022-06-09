import numpy as np



class hashtag:
    def __init__(self):
        self.hash_table = np.zeros(256).astype(int)
        self.hash = 0

    def pearson_hash(self, x):
        #print(type(x), type(self.hash))
        idx = x^self.hash
        self.hash = self.hash_table[idx]

        print(x, idx, self.hash)

    def hashtag(self, x, mode, idx):

        if (mode == 0):
            self.hash = x
        elif (mode == 1):
            self.hash_table[idx] = x
            #print(self.hash_table[idx])
        elif (mode == 2):
            self.pearson_hash(x)
        elif (mode == 3):
            return self.hash
        return 0


def main(layer):
    Hashtag_ = hashtag()

    fp1 = open('hash_table.dat', 'r')
    lines = fp1.readlines()

    Hashtag_.hashtag(0, 0, 0)

    mode = 1
    i = 0
    for line in lines:
        
        val = line.strip()
        Hashtag_.hashtag(int(float(val)), mode, i)
        i += 1
    fp1.close()

    mode = 2
    fp2 = open(f'{layer}.dat', 'r')
    lines = fp2.readlines()
    i = 0

    for line in lines:
        val = line.strip()
        if i >20:
            break
        print(float(val))
        Hashtag_.hashtag(int(float(val)*1000), mode, 0)
        i += 1
    fp2.close()

    mode = 3
    hash = Hashtag_.hashtag(int(float(val)*1000), mode, 0)
    print("final hash:", hash)

if __name__ == '__main__':
	main()
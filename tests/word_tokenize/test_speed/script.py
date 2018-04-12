from underthesea import word_tokenize
import time

start = time.time()
infile = "content_1k.txt"
outfile = "tokenized_content_1k.txt"
tf = open(outfile, "a")
with open(infile) as f:
    for i, line in enumerate(f):
        content = " ".join(line.split("\t")[1:])
        tokenized_content = word_tokenize(content, format="text")
        # tf.write(tokenized_content)
        # tf.write("\n")
        if i % 100 == 0 and i > 0:
            print(i)
end = time.time()
print(end - start)

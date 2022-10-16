from collections import namedtuple
data_dir = 'arabicRawData/'
Sample = namedtuple('Sample', 'gt_text, file_path')
samples = []
chars = set()
f = open(data_dir + 'gt/words.txt')

bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
for line in f:
    line = line.split(',')  # ignore empty and comment lines
    if not line or line[0] == '#':
        continue
    path = line[0]
    text = line[1][:-1]
    chars = chars.union(set(list(text)))
    path = data_dir +path







    # put sample into list
    samples.append(Sample(text, path))
print(samples[3])
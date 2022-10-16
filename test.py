from collections import namedtuple
data_dir = 'data/dataset/'
Sample = namedtuple('Sample', 'gt_text, file_path')
samples = []
chars = set()
f = open(data_dir + 'gt/words.txt')

bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset
for line in f:
    # ignore empty and comment lines
    line = line.strip()
    if not line or line[0] == '#':
        continue

    line_split = line.split(' ')
    assert len(line_split) >= 9

    # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
    file_name_split = line_split[0].split('-')
    file_name_subdir1 = file_name_split[0]
    file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
    file_base_name = line_split[0] + '.png'
    file_name = data_dir +'img/'+ file_name_subdir1 +'/'+ file_name_subdir2 +'/'+ file_base_name

    if line_split[0] in bad_samples_reference:
        print('Ignoring known broken image:', file_name)
        continue

    # GT text are columns starting at 9
    gt_text = ' '.join(line_split[8:])
    chars = chars.union(set(list(gt_text)))

    # put sample into list
    samples.append(Sample(gt_text, file_name))
print(samples[0])
import os
import pathlib
import sys

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))



def write_samples(samples, file_path, opt='w'):
    """Write the samples into a file.

    Args:
        samples (list): The list of samples to write.
        file_path (str): The path of file to write.
        opt (str, optional): The "mode" parameter in open(). Defaults to 'w'.
    """
    with open(file_path, opt, encoding='utf-8') as file:
        for line in samples:
            file.write(line)
            file.write('\n')

def partition(samples):
    train, dev, test = [], [], []
    count = 0
    for sample in samples:
        count += 1
        if count % 1000 == 0:
            print(count)
        if count <= 1000:
            test.append(sample)
        elif count <= 6000:
            dev.append(sample)
        else:
            train.append(sample)
    print('train: ', len(train))

    write_samples(train, os.path.join(abs_path, "../files/train.txt"))
    write_samples(dev, os.path.join(abs_path, "../files/dev.txt"))
    write_samples(test, os.path.join(abs_path, "../files/test.txt"))


import copy
import torch
import random

def is_sorted(index_list):
    return all([index_list[i] <= index_list[i + 1] for i in range(len(index_list) - 1)])

def rearrange(X, slice_nums, times):
    initial_labels = torch.ones((X.size()[0]), dtype=int)

    each_data_length = X.size()[-1]
    block_length = each_data_length // slice_nums

    temp_X_part = []
    for j in range(slice_nums):
        if each_data_length - (j + 1) * block_length < block_length:
            index = slice(j * block_length, each_data_length)
        else:
            index = slice(j * block_length, (j + 1) * block_length)
        temp_X_part.append(X[:, :, index])

    final_X, final_y = None, None

    for _ in range(times):
        index = [i for i in range(len(temp_X_part))]
        random.shuffle(index)

        while is_sorted(index):
            random.shuffle(index)

        build_X = None
        for j in index:
            if build_X is None:
                build_X = temp_X_part[j]
            else:
                build_X = torch.cat((build_X, temp_X_part[j]), dim=2)

        build_labels = torch.zeros((build_X.size()[0]), dtype=int)

        if final_X is None:
            final_X = torch.cat((X, build_X), dim=0)
        else:
            final_X = torch.cat((final_X, build_X), dim=0)

        if final_y is None:
            final_y = torch.cat((initial_labels, build_labels), dim=0)
        else:
            final_y = torch.cat((final_y, build_labels), dim=0)

    # print(final_X.size())
    # print(final_y.size())

    final_index = [i for i in range(len(final_X))]
    random.shuffle(final_index)

    while is_sorted(final_index):
        random.shuffle(final_index)

    return final_X, final_y, final_index
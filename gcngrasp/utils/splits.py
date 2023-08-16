import os
import sys
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../../'))
from visualize import mkdir

# DO NOT CHANGE THIS
TRAIN_SPLIT = 0.65
TEST_SPLIT = 0.25
VAL_SPLIT = 0.10


def read_txt_file_lines(path_txt_file):
    txt_file = open(path_txt_file, 'r')
    lines = txt_file.readlines()
    txt_file.close()
    return lines


def write_txt_file_lines(path_txt_file, lines):
    assert isinstance(lines, list)
    txt_file = open(path_txt_file, 'w')
    txt_file.writelines(lines)
    txt_file.close()
    return True


def parse_line(line):
    assert isinstance(line, str)
    line = line.split('\n')[0]
    try:
        (data_dsc, label) = line.split(':')
    except BaseException:
        embed()
    label = int(label)
    label = label == 1

    obj, grasp_id, task = data_dsc.split('-')
    obj = str(obj)
    grasp_id = int(grasp_id)
    task = str(task)
    obj_class = obj[obj.find('_') + 1:]
    return obj, obj_class, grasp_id, task, label


def get_ot_pairs_taskgrasp(task1_results_file):
    lines = read_txt_file_lines(task1_results_file)
    object_task_pairs = defaultdict(list)
    for line in lines:
        assert isinstance(line, str)
        line = line.split('\n')[0]
        (obj_instance, task, label) = line.split('-')
        ot_pair = "{}-{}".format(obj_instance, task)
        object_task_pairs[label].append(ot_pair)
    object_task_pairs = dict(object_task_pairs)
    return object_task_pairs


def get_split_otg(lines, parse_func):
    assert TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT <= 1.0
    num_train = int(len(lines) * TRAIN_SPLIT)
    num_test = int(len(lines) * TEST_SPLIT)
    num_val = int(len(lines) - num_train - num_test)
    idxes = list(range(len(lines)))
    np.random.shuffle(idxes)
    train_idxs = idxes[:num_train]
    test_idxs = idxes[num_train:num_train + num_test]
    val_idxs = idxes[num_train + num_test:]

    lines = np.array(lines)
    lines_train = list(lines[train_idxs])
    lines_test = list(lines[test_idxs])
    lines_val = list(lines[val_idxs])
    print('Split mode Grasps (OTG): Train {} grasps, Test {}, Val {}'.format(
        len(lines_train), len(lines_test), len(lines_val)))
    return lines_train, lines_test, lines_val


def get_split_ot(lines, parse_func):
    assert TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT <= 1.0
    ot_pairs = []
    for line in lines:
        _, obj_class, _, task, _ = parse_func(line)
        ot = "{}_{}".format(obj_class, task)
        ot_pairs.append(ot)
    ot_pairs = list(set(ot_pairs))

    num_train = int(len(ot_pairs) * TRAIN_SPLIT)
    num_test = int(len(ot_pairs) * TEST_SPLIT)
    num_val = int(len(ot_pairs) - num_train - num_test)
    idxes = list(range(len(ot_pairs)))
    np.random.shuffle(idxes)
    train_idxs = idxes[:num_train]
    test_idxs = idxes[num_train:num_train + num_test]
    val_idxs = idxes[num_train + num_test:]

    ot_pairs = np.array(ot_pairs)
    ot_train = list(ot_pairs[train_idxs])
    ot_test = list(ot_pairs[test_idxs])
    ot_val = list(ot_pairs[val_idxs])

    lines_train = []
    lines_test = []
    lines_val = []

    for line in lines:
        _, obj_class, _, task, _ = parse_func(line)
        ot = "{}_{}".format(obj_class, task)
        if ot in ot_train:
            lines_train.append(line)
        elif ot in ot_test:
            lines_test.append(line)
        elif ot in ot_val:
            lines_val.append(line)
        else:
            raise ValueError("Invalid ot pair {}".format(ot))

    print(
        'Split mode Objectclass-Task: Train {} pairs/{} grasps, Test {}/{}, Val {}/{}'.format(
            len(ot_train),
            len(lines_train),
            len(ot_test),
            len(lines_test),
            len(ot_val),
            len(lines_val)))
    return lines_train, lines_test, lines_val, ot_train, ot_test, ot_val


def get_split_o(lines, parse_func):
    assert TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT <= 1.0
    obj_classses = []
    for line in lines:
        _, obj_class, _, _, _ = parse_func(line)
        obj_classses.append(obj_class)
    obj_classses = list(set(obj_classses))

    num_train = int(len(obj_classses) * TRAIN_SPLIT)
    num_test = int(len(obj_classses) * TEST_SPLIT)
    num_val = int(len(obj_classses) - num_train - num_test)
    idxes = list(range(len(obj_classses)))
    np.random.shuffle(idxes)
    train_idxs = idxes[:num_train]
    test_idxs = idxes[num_train:num_train + num_test]
    val_idxs = idxes[num_train + num_test:]

    obj_classses = np.array(obj_classses)
    o_train = list(obj_classses[train_idxs])
    o_test = list(obj_classses[test_idxs])
    o_val = list(obj_classses[val_idxs])

    lines_train = []
    lines_test = []
    lines_val = []

    for line in lines:
        _, obj_class, _, _, _ = parse_func(line)
        if obj_class in o_train:
            lines_train.append(line)
        elif obj_class in o_test:
            lines_test.append(line)
        elif obj_class in o_val:
            lines_val.append(line)
        else:
            raise ValueError("Invalid object class {}".format(obj_class))

    print(
        'Split mode Object class: Train {} pairs/{} grasps, Test {}/{}, Val {}/{}'.format(
            len(o_train),
            len(lines_train),
            len(o_test),
            len(lines_test),
            len(o_val),
            len(lines_val)))
    return lines_train, lines_test, lines_val, o_train, o_test, o_val


def get_split_o_crossvalidation(
        lines,
        parse_func,
        num_splits=4,
        map_obj2class=None):
    """
    Creates splits with held-out object classes
    """
    assert TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT <= 1.0
    obj_classses = []
    for line in lines:
        obj, _, _, _, _ = parse_func(line)
        obj_class = map_obj2class[obj]
        obj_classses.append(obj_class)
    obj_classses = list(set(obj_classses))
    np.random.shuffle(obj_classses)
    num_classes = len(obj_classses)
    obj_classses = np.array(obj_classses)

    num_train = int(num_classes * TRAIN_SPLIT)
    num_test = int(num_classes * TEST_SPLIT)
    num_val = int(num_classes - num_train - num_test)
    idxes = list(range(num_classes))
    np.random.shuffle(idxes)
    all_idxes = list(range(num_classes))

    splits = {}
    num_object_classes_per_split = int(num_classes / num_splits)
    split_idx_counter = 0
    for i in range(0, num_classes, num_object_classes_per_split):

        if len(idxes[i:i + num_object_classes_per_split]
               ) < num_object_classes_per_split:
            break
        if split_idx_counter == num_splits - 1:
            test_idxs = idxes[i:]
        else:
            test_idxs = idxes[i:i + num_object_classes_per_split]

        train_val_idxs = list(set(all_idxes) - set(test_idxs))
        np.random.shuffle(train_val_idxs)

        val_idxs = train_val_idxs[:num_val]
        train_idxs = train_val_idxs[num_val:]

        assert set(test_idxs) | set(val_idxs) | set(
            train_idxs) == set(all_idxes)

        o_train = list(obj_classses[train_idxs])
        o_test = list(obj_classses[test_idxs])
        o_val = list(obj_classses[val_idxs])

        lines_train = []
        lines_test = []
        lines_val = []

        for line in lines:
            obj, _, _, _, _ = parse_func(line)
            obj_class = map_obj2class[obj]
            if obj_class in o_train:
                lines_train.append(line)
            elif obj_class in o_test:
                lines_test.append(line)
            elif obj_class in o_val:
                lines_val.append(line)
            else:
                raise ValueError("Invalid object class {}".format(obj_class))

        print(
            'Split mode Objects, Index {}: Train {} elements/{} grasps, Test {}/{}, Val {}/{}'.format(
                split_idx_counter,
                len(o_train),
                len(lines_train),
                len(o_test),
                len(lines_test),
                len(o_val),
                len(lines_val)))

        splits[split_idx_counter] = [
            lines_train,
            lines_test,
            lines_val,
            o_train,
            o_test,
            o_val]
        split_idx_counter += 1

    return splits


def get_split_i_crossvalidation(lines, parse_func, num_splits=4):
    """
    Creates splits with held-out object instances
    """
    assert TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT <= 1.0
    obj_instances = []
    for line in lines:
        obj, _, _, _, _ = parse_func(line)
        obj_instances.append(obj)
    obj_instances = list(set(obj_instances))
    np.random.shuffle(obj_instances)
    num_classes = len(obj_instances)
    obj_instances = np.array(obj_instances)

    num_train = int(num_classes * TRAIN_SPLIT)
    num_test = int(num_classes * TEST_SPLIT)
    num_val = int(num_classes - num_train - num_test)
    idxes = list(range(num_classes))
    np.random.shuffle(idxes)
    all_idxes = list(range(num_classes))

    splits = {}
    num_object_classes_per_split = int(num_classes / num_splits)
    split_idx_counter = 0
    for i in range(0, num_classes, num_object_classes_per_split):

        if len(idxes[i:i + num_object_classes_per_split]
               ) < num_object_classes_per_split:
            break
        if split_idx_counter == num_splits - 1:
            test_idxs = idxes[i:]
        else:
            test_idxs = idxes[i:i + num_object_classes_per_split]

        train_val_idxs = list(set(all_idxes) - set(test_idxs))
        np.random.shuffle(train_val_idxs)

        val_idxs = train_val_idxs[:num_val]
        train_idxs = train_val_idxs[num_val:]

        assert set(test_idxs) | set(val_idxs) | set(
            train_idxs) == set(all_idxes)

        i_train = list(obj_instances[train_idxs])
        i_test = list(obj_instances[test_idxs])
        i_val = list(obj_instances[val_idxs])

        lines_train = []
        lines_test = []
        lines_val = []

        for line in lines:
            obj, _, _, _, _ = parse_func(line)
            if obj in i_train:
                lines_train.append(line)
            elif obj in i_test:
                lines_test.append(line)
            elif obj in i_val:
                lines_val.append(line)
            else:
                raise ValueError(
                    "Invalid object instances {}".format(obj_class))

        print(
            'Split mode Instances, Index {}: Train {} elements/{} grasps, Test {}/{}, Val {}/{}'.format(
                split_idx_counter,
                len(i_train),
                len(lines_train),
                len(i_test),
                len(lines_test),
                len(i_val),
                len(lines_val)))

        splits[split_idx_counter] = [
            lines_train,
            lines_test,
            lines_val,
            i_train,
            i_test,
            i_val]
        split_idx_counter += 1

    return splits


def get_split_t(lines, parse_func):
    assert TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT <= 1.0
    tasks = []
    for line in lines:
        _, _, _, task, _ = parse_func(line)
        tasks.append(task)
    tasks = list(set(tasks))

    num_train = int(len(tasks) * TRAIN_SPLIT)
    num_test = int(len(tasks) * TEST_SPLIT)
    num_val = int(len(tasks) - num_train - num_test)
    idxes = list(range(len(tasks)))
    np.random.shuffle(idxes)
    train_idxs = idxes[:num_train]
    test_idxs = idxes[num_train:num_train + num_test]
    val_idxs = idxes[num_train + num_test:]

    tasks = np.array(tasks)
    t_train = list(tasks[train_idxs])
    t_test = list(tasks[test_idxs])
    t_val = list(tasks[val_idxs])

    lines_train = []
    lines_test = []
    lines_val = []

    for line in lines:
        _, _, _, task, _ = parse_func(line)
        if task in t_train:
            lines_train.append(line)
        elif task in t_test:
            lines_test.append(line)
        elif task in t_val:
            lines_val.append(line)
        else:
            raise ValueError("Invalid task {}".format(obj_class))

    print(
        'Split mode Tasks: Train {} pairs/{} grasps, Test {}/{}, Val {}/{}'.format(
            len(t_train),
            len(lines_train),
            len(t_test),
            len(lines_test),
            len(t_val),
            len(lines_val)))
    return lines_train, lines_test, lines_val, t_train, t_test, t_val


def get_split_t_crossvalidation(lines, parse_func, num_splits=4):
    """
    Creates splits with held-out tasks
    """
    assert TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT <= 1.0
    tasks = []
    for line in lines:
        _, _, _, task, _ = parse_func(line)
        tasks.append(task)
    tasks = list(set(tasks))
    np.random.shuffle(tasks)
    num_tasks = len(tasks)
    tasks = np.array(tasks)

    num_train = int(num_tasks * TRAIN_SPLIT)
    num_test = int(num_tasks * TEST_SPLIT)
    num_val = int(num_tasks - num_train - num_test)
    idxes = list(range(num_tasks))
    np.random.shuffle(idxes)
    all_idxes = list(range(num_tasks))

    splits = {}
    num_tasks_per_split = int(num_tasks / num_splits)
    split_idx_counter = 0
    for i in range(0, num_tasks, num_tasks_per_split):

        if len(idxes[i:i + num_tasks_per_split]) < num_tasks_per_split:
            break
        if split_idx_counter == num_splits - 1:
            test_idxs = idxes[i:]
        else:
            test_idxs = idxes[i:i + num_tasks_per_split]

        train_val_idxs = list(set(all_idxes) - set(test_idxs))
        np.random.shuffle(train_val_idxs)

        val_idxs = train_val_idxs[:num_val]
        train_idxs = train_val_idxs[num_val:]

        assert set(test_idxs) | set(val_idxs) | set(
            train_idxs) == set(all_idxes)

        t_train = list(tasks[train_idxs])
        t_test = list(tasks[test_idxs])
        t_val = list(tasks[val_idxs])

        lines_train = []
        lines_test = []
        lines_val = []

        for line in lines:
            _, _, _, task, _ = parse_func(line)
            if task in t_train:
                lines_train.append(line)
            elif task in t_test:
                lines_test.append(line)
            elif task in t_val:
                lines_val.append(line)
            else:
                raise ValueError("Invalid task {}".format(obj_class))

        print(
            'Split mode Tasks, Index {}: Train {} elements/{} grasps, Test {}/{}, Val {}/{}'.format(
                split_idx_counter,
                len(t_train),
                len(lines_train),
                len(t_test),
                len(lines_test),
                len(t_val),
                len(lines_val)))

        splits[split_idx_counter] = [
            lines_train,
            lines_test,
            lines_val,
            t_train,
            t_test,
            t_val]
        split_idx_counter += 1

    return splits


def get_split_lines(lines, map_obj2class, split_items, split_mode):
    split_items = [item.split('\n')[0] for item in split_items]
    lines_filtered = []

    for line in lines:
        obj, obj_class, _, task, _ = parse_line(line)
        obj_class = map_obj2class[obj]

        if split_mode == 't':
            if task in split_items:
                lines_filtered.append(line)
        elif split_mode == 'o':
            if obj_class in split_items:
                lines_filtered.append(line)
        elif split_mode == 'i':
            if obj in split_items:
                lines_filtered.append(line)
        else:
            raise InvalidArgumentError(
                'Unknown split mode {}'.format(split_mode))

    return lines_filtered


def get_split_data(
    base_dir,
    folder_dir,
    train,
    split_mode,
    split_idx,
    split_version,
    use_task1_grasps,
    data_txt_splits,
    map_obj2class,
    parse_func,
    get_object_task_pairs,
    get_task1_hits_func
):
    """ Function to load train/test/val data based on splits.
    Loads the splits if they have been pre-generated or creates them otherwise
    """

    if split_version:
        split_dir = 'splits_final'
        assert use_task1_grasps

        if split_mode == 'i':
            heldout_txt_splits = {
                0: 'test_i.txt',
                1: 'train_i.txt',
                2: 'val_i.txt'}
        elif split_mode == 'o':
            heldout_txt_splits = {
                0: 'test_o.txt',
                1: 'train_o.txt',
                2: 'val_o.txt'}
        elif split_mode == 't':
            heldout_txt_splits = {
                0: 'test_t.txt',
                1: 'train_t.txt',
                2: 'val_t.txt'}
        elif split_mode in ['sg', 'si', 'so', 'st']:
            heldout_txt_splits = {
                0: 'test_i.txt',
                1: 'train_i.txt',
                2: 'val_i.txt'}
        else:
            raise NotImplementedError(
                "Split mode {} not implemented".format(split_mode))

    else:
        split_dir = 'splits_wtask1' if use_task1_grasps else 'splits'
    data_txt_dir = os.path.join(
        base_dir,
        folder_dir,
        split_dir,
        split_mode,
        str(split_idx))

    # Get Stage 1 results
    task1_results_file = os.path.join(
        base_dir, folder_dir, 'task1_results.txt')
    assert os.path.exists(task1_results_file)
    object_task_pairs = get_object_task_pairs(task1_results_file)

    data_txt_path = os.path.join(data_txt_dir, data_txt_splits[train])
    if not os.path.exists(data_txt_path):
        mkdir(data_txt_dir)

        annotations_txt_path = os.path.join(
            base_dir, folder_dir, "task2_results.txt")

        if not os.path.exists(annotations_txt_path):
            raise ValueError(
                "Annotations file not found {}".format(annotations_txt_path))

        if use_task1_grasps:
            lines_task2 = read_txt_file_lines(annotations_txt_path)
            lines_task2 = lines_task2[:-1]
            lines_task1 = get_task1_hits_func(object_task_pairs)
            lines = lines_task1 + lines_task2
            np.random.shuffle(lines)
        else:
            lines = read_txt_file_lines(annotations_txt_path)

        if split_version:

            num_splits = int(1.0 / TEST_SPLIT)
            print(
                'Generating CROSS-VALIDATION data splits for type {} ...'.format(split_mode))
            if split_mode == 'g':
                for split_idx in range(num_splits):
                    lines_train, lines_test, lines_val = get_split_otg(
                        copy.deepcopy(lines), parse_func)
                    data_txt_dir = os.path.join(
                        base_dir, folder_dir, split_dir, split_mode, str(split_idx))
                    mkdir(data_txt_dir)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "train_split.txt"),
                        lines_train)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "test_split.txt"),
                        lines_test)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "val_split.txt"),
                        lines_val)
            elif split_mode == 'ot':
                raise NotImplementedError()
            elif split_mode == 'i':
                splits_i = get_split_i_crossvalidation(
                    lines, parse_func, num_splits=num_splits)

                for idx in splits_i:
                    lines_train, lines_test, lines_val, i_train, i_test, i_val = splits_i[idx]
                    data_txt_dir = os.path.join(
                        base_dir, folder_dir, split_dir, split_mode, str(idx))
                    mkdir(data_txt_dir)
                    i_train = [elem + '\n' for elem in i_train]
                    i_test = [elem + '\n' for elem in i_test]
                    i_val = [elem + '\n' for elem in i_val]
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "train_i.txt"),
                        i_train)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "test_i.txt"),
                        i_test)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "val_i.txt"),
                        i_val)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "train_split.txt"),
                        lines_train)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "test_split.txt"),
                        lines_test)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "val_split.txt"),
                        lines_val)

            elif split_mode == 'o':
                splits_o = get_split_o_crossvalidation(
                    lines, parse_func, num_splits=num_splits, map_obj2class=map_obj2class)

                for idx in splits_o:
                    lines_train, lines_test, lines_val, o_train, o_test, o_val = splits_o[idx]
                    data_txt_dir = os.path.join(
                        base_dir, folder_dir, split_dir, split_mode, str(idx))
                    mkdir(data_txt_dir)
                    o_train = [elem + '\n' for elem in o_train]
                    o_test = [elem + '\n' for elem in o_test]
                    o_val = [elem + '\n' for elem in o_val]
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "train_o.txt"),
                        o_train)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "test_o.txt"),
                        o_test)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "val_o.txt"),
                        o_val)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "train_split.txt"),
                        lines_train)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "test_split.txt"),
                        lines_test)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "val_split.txt"),
                        lines_val)
            elif split_mode == 't':
                splits_t = get_split_t_crossvalidation(
                    lines, parse_func, num_splits=num_splits)

                for idx in splits_t:
                    lines_train, lines_test, lines_val, t_train, t_test, t_val = splits_t[idx]
                    data_txt_dir = os.path.join(
                        base_dir, folder_dir, split_dir, split_mode, str(idx))
                    mkdir(data_txt_dir)
                    t_train = [elem + '\n' for elem in t_train]
                    t_test = [elem + '\n' for elem in t_test]
                    t_val = [elem + '\n' for elem in t_val]
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "train_t.txt"),
                        t_train)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "test_t.txt"),
                        t_test)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "val_t.txt"),
                        t_val)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "train_split.txt"),
                        lines_train)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "test_split.txt"),
                        lines_test)
                    write_txt_file_lines(
                        os.path.join(
                            data_txt_dir,
                            "val_split.txt"),
                        lines_val)
            else:
                raise InvalidArgumentError(
                    'Unknown split mode {}'.format(split_mode))

        else:
            print('Generating RANDOM data splits for type {} ...'.format(split_mode))

            if split_mode == 'otg':
                lines_train, lines_test, lines_val = get_split_otg(
                    lines, parse_func)
            elif split_mode == 'ot':
                lines_train, lines_test, lines_val, ot_train, ot_test, ot_val = get_split_ot(
                    lines, parse_func)
                ot_train = [elem + '\n' for elem in ot_train]
                ot_test = [elem + '\n' for elem in ot_test]
                ot_val = [elem + '\n' for elem in ot_val]
                write_txt_file_lines(
                    os.path.join(
                        data_txt_dir,
                        "train_ot.txt"),
                    ot_train)
                write_txt_file_lines(
                    os.path.join(
                        data_txt_dir,
                        "test_ot.txt"),
                    ot_test)
                write_txt_file_lines(
                    os.path.join(
                        data_txt_dir,
                        "val_ot.txt"),
                    ot_val)
            elif split_mode == 'o':
                lines_train, lines_test, lines_val, o_train, o_test, o_val = get_split_o(
                    lines, parse_func)
                o_train = [elem + '\n' for elem in o_train]
                o_test = [elem + '\n' for elem in o_test]
                o_val = [elem + '\n' for elem in o_val]
                write_txt_file_lines(
                    os.path.join(
                        data_txt_dir,
                        "train_o.txt"),
                    o_train)
                write_txt_file_lines(
                    os.path.join(
                        data_txt_dir,
                        "test_o.txt"),
                    o_test)
                write_txt_file_lines(
                    os.path.join(
                        data_txt_dir,
                        "val_o.txt"),
                    o_val)
            elif split_mode == 't':
                lines_train, lines_test, lines_val, t_train, t_test, t_val = get_split_t(
                    lines, parse_func)
                t_train = [elem + '\n' for elem in t_train]
                t_test = [elem + '\n' for elem in t_test]
                t_val = [elem + '\n' for elem in t_val]
                write_txt_file_lines(
                    os.path.join(
                        data_txt_dir,
                        "train_t.txt"),
                    t_train)
                write_txt_file_lines(
                    os.path.join(
                        data_txt_dir,
                        "test_t.txt"),
                    t_test)
                write_txt_file_lines(
                    os.path.join(
                        data_txt_dir,
                        "val_t.txt"),
                    t_val)
            else:
                raise InvalidArgumentError(
                    'Unknown split mode {}'.format(split_mode))

            write_txt_file_lines(
                os.path.join(
                    data_txt_dir,
                    "train_split.txt"),
                lines_train)
            write_txt_file_lines(
                os.path.join(
                    data_txt_dir,
                    "test_split.txt"),
                lines_test)
            write_txt_file_lines(
                os.path.join(
                    data_txt_dir,
                    "val_split.txt"),
                lines_val)
    else:
        print('Loading from pregenerated data split...')

    # Do not remove this line
    data_txt_dir = os.path.join(
        base_dir,
        folder_dir,
        split_dir,
        split_mode,
        str(split_idx))

    lines = read_txt_file_lines(data_txt_path)

    print('Number of lines:{}'.format(len(lines)))
    return lines

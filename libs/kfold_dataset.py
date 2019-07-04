import os
import sys
import shutil
import random


def refresh_k_fold_dataset(source, dest, nb_folds, dev_train_set_percent=90,
                           verbose=True):

    '''
    Sort files into "train", "dev" and "test" sets following a k-fold setup.

    This function copies the folder structure from the source directory into
    each of the sets in the destination. The files are split, per class, into a
    test list containing all the test images, and a dev_train list containing
    the rest.

    Each time the funcion is called the dev_train list is split and a different
    subset is used as the validation list, the rest being used for training.

    The function also checks if files present in the destination directory are
    no longer present in the corresponding list and viceversa, to respectively
    remove or add the needed files only, to avoid redundant operations.

    Visual explanation:

        - input folder              -output folder
            -class1                     -train
            -...                            -class1
            -classN                         -...
                                            -classN
                                        -dev
                                            -class1
                                            -...
                                            -classN
                                        -test (constant throughout iterations)
                                            -class1
                                            -...
                                            -classN

    Args:
        source: Input directory containing the classified images.
        dest: Output directory to be created containig the sorted
            sets.
        nb_folds: Number of folds
        dev_train_set_percent: Percentage of files to be sorted into the
            "dev_train" set which will be used for the validation and trainig
            sets. The rest is used as test files. This is set to 90 by default.
        verbose: Prints the number of files processed so far. True by default.

    '''

    sets = ["test", "dev", "train"]

    if not hasattr(refresh_k_fold_dataset, "iteration"):
        refresh_k_fold_dataset.iteration = 0
        # Create dest directory
        create_dir(dest)
        # Create sets with the correct folder structure
        for set_name in sets:
            dest_sub_dir = os.path.join(dest, set_name)
            shutil.copytree(source, dest_sub_dir, copy_function=ignoreFiles)

    # Calculate percentages
    test_set_percent = 100 - dev_train_set_percent
    dev_set_percent = dev_train_set_percent // nb_folds

    # Find the number of files to convert
    num_images_total = get_nb_files(source)

    random.seed(1)  # Fixed seed to keep sets the same between executions
    num_images_processed = 0
    num_dirs_walked = 0

    for source_dir_path, _, images in os.walk(source):
        if num_dirs_walked > 0:    # Ignore the first "root" directory
            # Order and shuffle the images to have a random list per class
            images.sort()
            random.shuffle(images)
            # Create test, dev and train lists for the currrent class
            nb_images_class = len(images)
            nb_images_test = nb_images_class * test_set_percent // 100
            nb_images_dev = nb_images_class * dev_set_percent // 100
            images_test = images[-nb_images_test:]
            images_dev_train = images[:(nb_images_class-nb_images_test)]
            dev_start = nb_images_dev * refresh_k_fold_dataset.iteration
            images_dev = images_dev_train[dev_start:dev_start+nb_images_dev]
            images_train = [i for i in images_dev_train if i not in images_dev]
            image_sets = {"test": images_test,
                          "dev": images_dev,
                          "train": images_train}

            for set_name in sets:
                # Create the correct output sub directory path
                source_sub_dir = os.path.split(source_dir_path)[1]
                dest_sub_dir = os.path.join(dest, set_name)
                dest_sub_dir = os.path.join(dest_sub_dir, source_sub_dir)
                # If file in dir not in list, delete it
                dest_files = os.listdir(dest_sub_dir)
                images_delete = [i for i in dest_files if i not in
                                 image_sets[set_name]]
                for image in images_delete:
                    image_out = os.path.join(dest_sub_dir, image)
                    os.remove(image_out)
                # If file in list not in dir, copy it
                for i in range(len(image_sets[set_name])):
                    image_out = os.path.join(dest_sub_dir,
                                             image_sets[set_name][i])
                    image_in = os.path.join(source_dir_path,
                                            image_sets[set_name][i])
                    if not os.path.isfile(image_out):
                        shutil.copyfile(image_in, image_out)
                    # Print progress info
                    num_images_processed += 1
                    if verbose:
                        print_progress(refresh_k_fold_dataset.iteration,
                                       num_images_processed, num_images_total)

        num_dirs_walked += 1
    if verbose:
        print("")
    refresh_k_fold_dataset.iteration += 1


def ignoreFiles(_, __): pass


def create_dir(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def get_nb_files(directory):
    nb_files = 0
    for _, _, images in os.walk(directory):
        nb_files += len(images)
    return nb_files


def print_progress(fold, processed, total):
    print(f"Fold: {fold} Processed {processed}/{total} images",
          f"({(processed*100/total):4.2f} %)", end='\r')

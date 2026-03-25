import numpy as np


def oversample_dataset(dataset, oversampling_ratio=0.7):
    """
    This function oversamples the classes that occur less frequently in the dataset.
    The occurence of each class is counted and each class is oversampled 70% of the difference between the most common class and the class in question.
    """

    class_occurences = dataset[:]['labels'].numpy()
    unique_classes, counts = np.unique(class_occurences, return_counts=True)
    max_count = counts.max()
    indices_with_oversamples = []
    for class_idx, count in zip(unique_classes, counts):
        class_indices = np.where(class_occurences == class_idx)[0]
        indices_with_oversamples.extend(class_indices)
        oversample_count = int(oversampling_ratio * (max_count - count))
        indices_with_oversamples.extend(np.random.choice(class_indices, oversample_count))
    
    return indices_with_oversamples
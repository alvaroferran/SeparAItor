def get_most_frequent(preds):
    return max(set(preds), key=preds.count)


def sort_item(item):
    PMD = ["plastic", "cans", "cartons"]
    organic = ["eggs"]
    glass = ["glass"]
    paper = ["paper"]
    classes = [PMD, organic, glass, paper]
    for class_item in classes:
        if item in class_item:
            return classes.index(class_item)
    return None

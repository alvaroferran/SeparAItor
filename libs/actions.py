def get_most_frequent(preds):
    return max(set(preds), key=preds.count)


def sort_item(item):
    PMC = ["plastic", "cans", "cartons"]
    organic = ["eggs"]
    glass = ["glass"]
    paper = ["paper"]
    classes = [glass, organic, PMC, paper]
    for class_item in classes:
        if item in class_item:
            return classes.index(class_item)
    return None

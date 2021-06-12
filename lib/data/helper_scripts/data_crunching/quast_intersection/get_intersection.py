base_path = "E:\\Users\\Documents\\School_Year_18-19\\Term_1\\CPSC_449\\Sealer_NN\\lib\\data\\helper_scripts\\data_crunching\\quast_intersection\\"

gappadder_set1 = open(base_path + "gappadder_set1.txt", "r")
sealer_set1 = open(base_path + "sealer_set1.txt", "r")
left_fixed = open(base_path + "left_fixed.txt", "r")
right_fixed = open(base_path + "right_fixed.txt", "r")

def to_list(file):
    list = []
    for id in file:
        list.append(id.rstrip("\n"))
    return list

def to_txt(iterable, file_path):
    file = open(file_path, "w+")
    length = len(iterable)
    acc = 1
    for id in iterable:
        if acc == length:
            file.write(id)
        else:
            file.write(id + "\n")
        acc += 1
    file.close()

gappadder_list = to_list(gappadder_set1)
sealer_list = to_list(sealer_set1)
gappredict_left_list = to_list(left_fixed)
gappredict_right_list = to_list(right_fixed)

gappadder_set = set(gappadder_list)
sealer_set = set(sealer_list)
gappredict_left_set = set(gappredict_left_list)
gappredict_right_set = set(gappredict_right_list)
gappredict_set = set(gappredict_left_list + gappredict_right_list)

intersection_set_left = gappadder_set.intersection(sealer_set).intersection(gappredict_left_set)
intersection_set_right = gappadder_set.intersection(sealer_set).intersection(gappredict_right_set)
intersection_set_overall = gappadder_set.intersection(sealer_set).intersection(gappredict_set)

to_txt(intersection_set_left, base_path + "left_intersection.txt")
to_txt(intersection_set_right, base_path + "right_intersection.txt")
to_txt(intersection_set_overall, base_path + "intersection.txt")

gappadder_set1.close()
sealer_set1.close()
left_fixed.close()
right_fixed.close()

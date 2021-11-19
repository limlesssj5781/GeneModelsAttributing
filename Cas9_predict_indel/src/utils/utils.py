#
def load_data(file):
    data = []
    with open(file,"r") as f:
        for line in f:
            line = line.split()
            data.append((line[0], float(line[1])))
    return data
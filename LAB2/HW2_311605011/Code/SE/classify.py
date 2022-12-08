import os

os.mkdir("gt_labels")
all_val = os.listdir("val_labels")
#print(all_val)
d = []
for v in all_val:
    a = os.path.join("val_labels", v)
    b = os.path.join("gt_labels", v)
    writer = open(b, 'w')
    print(a)
    with open (a, "r") as f:
        for line in f.readlines():
            line = line.split(" ")
            line[4] = line[4][:-1]
            #print(line)
            j = 0
            for i in line:
                line[j] = float(i)
                #print(i)
                j += 1

            x = [int(line[0]), int(1920 * (line[1] - 0.5 * line[3])), int(1080 * (line[2] - 0.5 * line[4])), int(1920 * (line[1] + 0.5 * line[3])), int(1080 * (line[2] + 0.5 * line[4]))]
            writer.write((str(x[0]) + ' '))
            writer.write((str(x[1]) + ' '))
            writer.write((str(x[2]) + ' '))
            writer.write((str(x[3]) + ' '))
            writer.write((str(x[4]) + ' \n'))
            #print(x)
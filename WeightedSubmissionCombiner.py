import csv
import pandas as pd

sub_files = [
    '../input/flower/submission_template.csv',
    '../input/flower/submission_template2.csv',
    '../input/flower2/submission_template3.csv',
]

# Weights of the individual subs
sub_weight = [
    0.734**2,
    0.779**2,
    0.781**2,
]

Hlabel = 'filename'
Htarget = 'category'
npt = 1
place_weights = {}
for i in range(npt):
    place_weights[i] = (1 / (i + 1))

print(place_weights)

lg = len(sub_files)
sub = [None] * lg
for i, file in enumerate(sub_files):
    print("Reading {}: w={} - {}".format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file, "r"))
    sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))

out = open("submission.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel, Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt, 0) + (place_weights[ind] * sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row[Hlabel], " ".join(tops_trgt)])
out.close()

import csv
import matplotlib.pyplot as plt

timestamps = []
scores = []

with open("log.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        if len(row) == 3:
            timestamps.append(float(row[0]))
            scores.append(int(row[2]))

plt.plot(timestamps, scores, marker='o')
plt.title("Suspicion Score Over Time")
plt.xlabel("Time")
plt.ylabel("Score")
plt.grid()

plt.show()
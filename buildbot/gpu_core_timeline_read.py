import json

import matplotlib.pyplot as plt

# Load the JSON file
with open("gpu_core_timeline_profilier_test.json") as f:
    data = json.load(f)

# Create a figure and axis
fig, ax = plt.subplots(dpi=200)


lane_count = 0
for d in data:
    lane_count = max(lane_count, d["lane"] + 1)


lane_loc_id = [0 for _ in range(lane_count)]


max_t_end = 0
for d in data:
    max_t_end = max(max_t_end, d["last_end"])

sum_t_blocks_first = 0
for d in data:
    sum_t_blocks_first += d["first_end"] - d["start"]

sum_t_blocks_last = 0
for d in data:
    sum_t_blocks_last += d["last_end"] - d["start"]


ker_per_sm = 16
width = 0.1


# Iterate over the array and plot bars
for index, item in enumerate(data):
    loc_lane_id_offset = lane_loc_id[item["lane"]] % ker_per_sm

    ax.bar(
        item["lane"] + (width) * (loc_lane_id_offset),
        (item["last_end"] - item["start"]) * 1e-9,
        bottom=item["start"] * 1e-9,
        width=width,
        color="blue",
        edgecolor="black",
    )
    ax.bar(
        item["lane"] + (width) * (loc_lane_id_offset),
        (item["first_end"] - item["start"]) * 1e-9,
        bottom=item["start"] * 1e-9,
        width=width / 1.5,
        color="red",
        edgecolor="black",
    )

    lane_loc_id[item["lane"]] += 1

# Set the x-axis label
ax.set_xlabel("Lane")

# Set the y-axis label
ax.set_ylabel("Time (s)")

ax.set_title("GPU Core Timeline (Update derivs)")

plt.savefig("update_derivs.png")

print("Total time (s) : ", max_t_end * 1e-9)
print(
    "Estimated ideal total time (s) (from last end) : ",
    sum_t_blocks_last * 1e-9 / (lane_count * ker_per_sm),
)
print(
    "Estimated ideal total time (s) (from first end) : ",
    sum_t_blocks_first * 1e-9 / (lane_count * ker_per_sm),
)

# Show the plot
plt.show()

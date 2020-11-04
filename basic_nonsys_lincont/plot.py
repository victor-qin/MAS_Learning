import numpy as np
import matplotlib.pyplot as plt
import json

# f = open("./basic_nonsys_lincont/records.json", 'r')
f = open("./basic_nonsys_lincont/records_103020_linsys.json", 'r')
# var = f.read()

struct = json.load(f)['agents']

plt.figure(1)
for t in struct:
    agent = t[list(t.keys())[0]]
    plt.plot(np.linspace(1, len(agent['error']), len(agent['error'])), agent['error'], label=list(t.keys())[0])

# CHANGE THE TITLE
plt.title("Single and Multiagent, Linear Controller Linear System")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.legend()
plt.show()


# plt
print("done")

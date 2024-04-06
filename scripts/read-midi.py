import mido
import matplotlib.pyplot as plt

class Note:
    def __init__(self, note, velocity, start, end):
        self.note = note
        self.velocity = velocity
        self.start = start
        self.end = end


SAMPLE = "Anno Domini - Hurting - 133 BPM F# Min Keys.mid"

mid = mido.MidiFile(f"../samples/{SAMPLE}", clip=True)
track = mid.tracks[0]

notes = [[] for _ in range(128)]

prev_time = 0
for msg in track:
    curr_time = msg.time + prev_time

    if msg.type == "note_on":
        notes[msg.note].append(Note(msg.note, msg.velocity, curr_time, curr_time))
    elif msg.type == "note_off":
        notes[msg.note][-1].end = curr_time

    prev_time += msg.time

for note in notes:
    for n in note:
        plt.plot([n.start, n.end], [n.note, n.note], color="black", linewidth=2)

plt.title(SAMPLE)
plt.xlabel("Time (ticks)")
plt.ylabel("Midi Note")

plt.savefig("./read-midi-output.png")

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import sounddevice as sd
import pretty_midi as pm
from midi2audio import FluidSynth

def load_data(path):
    path = "F.txt"

    # Load the data
    data = pd.read_csv(path, sep="\t", header=None)

    return data

def plot_data(data):
    # plot the data
    data.plot()
    plt.show()

def get_params(data):
    # Get the parameters
    symbolicLength = len(data)
    baseFreq = 440  # set base frequency (Hz) tuned to middle A over C, MIDI note value of 69
    sampleRate = 10000  # samples per second
    durationPerSymbol = 1 / 16  # in seconds. A "symbol" here means one entry in the voice vector
    ticksPerSymbol = math.floor(sampleRate * durationPerSymbol)

    return symbolicLength, baseFreq, sampleRate, durationPerSymbol, ticksPerSymbol

def get_soundvectors(data):
    
    symbolicLength, baseFreq, sampleRate, durationPerSymbol, ticksPerSymbol = get_params(data)

    soundvectors = []

    for voice_num in range(4):

        chosenVoice = voice_num
        voice = data.iloc[:, chosenVoice]

        soundvector = np.zeros((symbolicLength * ticksPerSymbol, 1))  # 2D array for compatibility
        currentSymbol = voice[0]  # Python indexing starts at 0
        startSymbolIndex = 0  # Python indexing adjustment

        for n in range(symbolicLength):
            if voice[n] != currentSymbol:
                stopSymbolIndex = n - 1  # Stop index adjustment for Python
                coveredSoundVectorIndices = np.arange(
                    startSymbolIndex * ticksPerSymbol,
                    (stopSymbolIndex + 1) * ticksPerSymbol
                )
                toneLength = len(coveredSoundVectorIndices)
                frequency = baseFreq * 2 ** ((currentSymbol - 69) / 12)
                toneVector = np.zeros((toneLength, 1))

                # Generate the sine wave
                for t in range(1, toneLength + 1):  # Python uses 0-based indexing
                    toneVector[t - 1, 0] = np.sin(2 * np.pi * frequency * t / sampleRate)

                # Assign toneVector to soundvector1
                soundvector[coveredSoundVectorIndices, 0] = toneVector.flatten()

                # Update variables for the next segment
                currentSymbol = voice[n]
                startSymbolIndex = n

        soundvectors.append(soundvector)

    return soundvectors


def play_soundvector(soundvectors):
    # Play the sound
    total_sound = (soundvectors[0] + soundvectors[1] + soundvectors[2] + soundvectors[3]) / 4
    total_sound = total_sound.flatten()

    # clip the sound to avoid distortion
    total_sound[total_sound > 0.98] = 0.98
    total_sound[total_sound < -0.98] = -0.98

    sd.play(total_sound, 10000)
    sd.wait()

def get_midi(data):
    # Save the sound to a MIDI file using pretty_midi
    cello = pm.Instrument(program=42)  # Program 42 is a cello in General MIDI

    symbolicLength, baseFreq, sampleRate, durationPerSymbol, ticksPerSymbol = get_params(data)

    cellos = []
    for voice_num in range(4):
        n = 0
        start = n * durationPerSymbol
        end = (n + 1) * durationPerSymbol
        for n in range(1, symbolicLength):
            if data.iloc[n, voice_num] == data.iloc[n - 1, voice_num]:
                end += durationPerSymbol
            else:
                note_velocity = 100
                if data.iloc[n - 1, voice_num] == 0:
                    note_velocity = 0
                pitch = data.iloc[n - 1, voice_num]
                cello.notes.append(pm.Note(
                    velocity=note_velocity,
                    pitch=pitch,
                    start=start,
                    end=end
                ))
                start = end
                end += durationPerSymbol
        cellos.append(cello)

    return cellos

def write_midi(cellos):
    pm_obj = pm.PrettyMIDI()

    # Add all cellos to the PrettyMIDI object so they are played simultaneously
    for cello in cellos:
        pm_obj.instruments.append(cello)

    # Write the MIDI file
    pm_obj.write("input_data_sound/cellos.mid")

def midi_to_wav():
    # Convert the MIDI file to a WAV file using FluidSynth
    fs = FluidSynth("soundfonts/Roland_SC-88.sf2")
    fs.midi_to_audio("input_data_sound/cellos.mid", "input_data_sound/cellos.wav")

def main():
    path = "F.txt"
    data = load_data(path)
    plot_data(data)
    # soundvectors = get_soundvectors(data)
    # play_soundvector(soundvectors)
    cellos = get_midi(data)
    write_midi(cellos)
    midi_to_wav()

if __name__ == "__main__":
    main()
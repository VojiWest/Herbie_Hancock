import pretty_midi as pm
import math
from midi2audio import FluidSynth

def get_params(data):
    # Get the parameters
    symbolicLength = len(data)
    baseFreq = 440  # set base frequency (Hz) tuned to middle A over C, MIDI note value of 69
    sampleRate = 10000  # samples per second
    durationPerSymbol = 1 / 16  # in seconds. A "symbol" here means one entry in the voice vector
    ticksPerSymbol = math.floor(sampleRate * durationPerSymbol)

    return symbolicLength, baseFreq, sampleRate, durationPerSymbol, ticksPerSymbol


def get_midi(data, instrument_num=42):
    # Save the sound to a MIDI file using pretty_midi
    cello = pm.Instrument(program=instrument_num)  # Program 42 is a cello in General MIDI

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


def write_midi(cellos, name):
    pm_obj = pm.PrettyMIDI()

    # Add all cellos to the PrettyMIDI object so they are played simultaneously
    for cello in cellos:
        pm_obj.instruments.append(cello)

    # Write the MIDI file
    pm_obj.write("Data Audio Outputs/"+name+".mid")


def midi_to_wav(name):
    # Convert the MIDI file to a WAV file using FluidSynth
    fs = FluidSynth("Soundfonts/Roland_SC-88.sf2")
    fs.midi_to_audio("Data Audio Outputs/"+name+".mid", "Data Audio Outputs/"+name+".wav")
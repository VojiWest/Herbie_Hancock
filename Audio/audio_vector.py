import numpy as np
import sounddevice as sd
import math

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
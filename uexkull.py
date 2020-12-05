import cv2
import numpy as np
from pydub import AudioSegment as audio

def magnitude(complex_val):
    return (complex_val.real**2 + complex_val.imag**2)**(1/2)

def normalize_wavelength(wavelength):
    '''
    "Normalize" wavelength to be in range 380 -> 750
    '''
    wavelength %= 750
    return max(380, wavelength) # has to be a better way!! >~<

def normalize_mat(mat):
    norm = np.zeros(mat.shape, np.uint8)
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            norm[row,col] = (mat[row,col] - min(mat)) / (max(mat) - min(mat))
    return norm

def wavelength_to_rgb(wavelength, gamma=0.8):
    '''
    Script to convert wavelength to rgb and frequency to wavelength.
    Adapted to python from R source here:
    https://gist.github.com/friendly/67a7df339aa999e2bcfcfec88311abfc
    '''
    if wavelength >= 380 & wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 & wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 & wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 & wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 & wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 & wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    return  magnitude(R) % 255, magnitude(G) % 255, magnitude(B) % 255

def frequency_to_wavelength(frequency):
    '''
    Made for sound waves: c ~= 343 m/s
    '''
    if frequency == 0:
        return 0
    else:
        return 343/frequency

def load_audio_data(song_file):
    # mp3 only for now
    byte_data = audio.from_mp3(song_file).raw_data
    int_data = np.array(list(byte_data), np.uint8)
    return int_data

def song_to_mat(int_data):

    # Get square matrix for image for data
    closest_square = int(np.floor(np.sqrt(len(int_data))))
    square_mat = np.zeros((closest_square, closest_square), np.uint8)

    # Copy over what you can
    for row in range(square_mat.shape[0]):
        for col in range(square_mat.shape[1]):
            square_mat[row,col] = int_data[row*col]

    return square_mat

def main():
    # Load audio data from file
    song_file_name = './Music/Math-Emo-Prog/Algernon Cadwallader/Summer Singles (2011)/04 (Na Na Na Na) Simulation.mp3'
    audio_data = load_audio_data(song_file_name)

    output_image_size = (500, 500)
    values_per_px = len(audio_data) // (output_image_size[0] * output_image_size[1])

    # Init image
    output_image = np.array(output_image_size, np.uint8)
    for i in range(0, len(audio_data), values_per_px):
        total = 0
        for j in range(values_per_px):
            total += audio_data[i+j]
        mean = total // values_per_px
        if i + values_per_px >= len(audio_data):
            break


    ## Convert data from array to square matrix
    #audio_data_mat = song_to_mat(audio_data)

    ## Perform DFT on data
    #dft_mat = np.array(np.fft.fft2(audio_data_mat), np.uint8)
    #dft_mat = np.fft.fftshift(dft_mat)

    #wavelength_data = np.zeros(dft_mat.shape, np.uint8)
    ##rgb_image = np.zeros([dft_mat.shape[0], dft_mat.shape[1], 3], np.uint8)

    ### Convert frequency to wavelength
    #for row in range(dft_mat.shape[0]):
    #    for col in range(dft_mat.shape[1]):
    #        wavelength_data[row,col] = frequency_to_wavelength(dft_mat[row,col])
    ##        #print(wavelength_to_rgb(wavelength_data[row,col]))
    ##        #rgb_image[row,col] = wavelength_to_rgb(normalize_wavelength(wavelength_data[row,col]))

    #result = normalize_mat(wavelength_data)

    #ifft = np.fft.ifftshift(result)
    #ifft = np.fft.ifft2(ifft)

    #print(ifft)
    #cv2.imshow('img',ifft)
    #cv2.imwrite('heyhey.jpg',ifft)
    #cv2.waitKey()
    #cv2.destroyWindow('img')


    ##cv2.imshow('dft_mat',dft_mat)
    ##cv2.imshow('wavelength_data',wavelength_data)
    ##cv2.imshow('rgb',rgb_image)
    ##cv2.waitKey()
    ##cv2.destroyWindow('dft_mat')
    ##cv2.destroyWindow('wavelength_data')
    ##cv2.destroyWindow('rgb')


if __name__ == "__main__":
    main()





























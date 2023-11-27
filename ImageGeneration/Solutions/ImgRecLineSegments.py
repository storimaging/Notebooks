import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import binom

#### Excercise 1 ####

def reconstructRectangles(LSDres, N, M):
    Im = np.zeros((N, M))  # Image of the detected rectangles
    X, Y = np.meshgrid(np.arange(0, M), np.arange(0, N))

    nrect, nc = LSDres.shape

    # iterate in detected rectangles from output of LSD algorithm.
    for k in range(nrect):
        # Get values from the rectangle
        x1 = LSDres[k, 0] # Coordinate x starting point
        y1 = LSDres[k, 1] # Coordinate y starting point
        x2 = LSDres[k, 2] # Coordinate x end point
        y2 = LSDres[k, 3] # Coordinate y end point
        w = LSDres[k, 4]  # Width rectangle

        # Checks if each point is inside the rectangle defined by (x1, y1) and (x2, y2).
        # The points that meet the condition are marked as True in the Im array.
        # Distance and orientation are check in Im1 and Im2 expresions.
        Im1 = np.abs((X - (x1 + x2) / 2) * (x2 - x1) + (Y - (y1 + y2) / 2) * (y2 - y1)) < 0.5 * ((x2 - x1)**2 + (y2 - y1)**2)
        Im2 = np.abs((X - (x1 + x2) / 2) * (y1 - y2) + (Y - (y1 + y2) / 2) * (x2 - x1)) < 0.5 * w * np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        Im += Im1 * Im2

    return Im

def visualize_rectangles(im):
    gray_image = im

    # make plot
    fig, ax = plt.subplots()

    # show image
    shw = ax.imshow(gray_image, interpolation='None', cmap='gray')

    # make bar
    bar = plt.colorbar(shw)

    plt.axis('off')
    plt.show()



#### Excercise 2 ####

def LSD_Statistics(LSDres, ImRect):
    # number of detected rectangles, and % of those detected with precision 1/8
    nb_precis0125 = np.sum(np.abs(LSDres[:, 5] - 0.125) <= 0.001)

    # % overlap of rectangles
    M,N = ImRect.shape
    overlap_rectangles_area = 100 * np.sum(ImRect > 1) / np.sum(ImRect > 0)
    overlap_total_area = 100 * np.sum(ImRect > 1) / (M * N)

    print("-------------------------------------------------")
    print("Number of detected rectangles: ", LSDres.shape[0])
    print("Percentage of overlap within area covered by rectangles: ", "{:.2f}".format(overlap_rectangles_area))
    print("Percentage of overlap with respect to the total number of pixels: ", "{:.2f}".format(overlap_total_area))
    print("Number of detected rectangles with precision 1/8: ", nb_precis0125)
    print("Percentage of detected rectangles with precision 1/8: ", 100 * nb_precis0125 / LSDres.shape[0])
    print("-------------------------------------------------")


#### Excercise 3 ####

def sampling_orientation_field(N,M,LSDres, Theta0):
    X, Y = np.meshgrid(np.arange(0, M), np.arange(0, N)) #Coordinate Matrix
    ImNew = 2 * np.pi * (np.random.rand(N, M) - 0.5)  # New image of orientations

    nrect, nc = LSDres.shape

    # iterate in detected rectangles from output of LSD algorithm.
    for r in range(nrect):
        # Get values from the rectangle
        x1 = LSDres[r, 0] # Coordinate x starting point
        y1 = LSDres[r, 1] # Coordinate y starting point
        x2 = LSDres[r, 2] # Coordinate x end point
        y2 = LSDres[r, 3] # Coordinate y end point
        w = LSDres[r, 4]  # Width rectangle
        precis = 0.125  # Fix precision to 1/8

        # Checks if each point is inside the rectangle defined by (x1, y1) and (x2, y2).
        # The points that meet the condition are marked as True in the ImRect array.
        # Distance and orientation are check in Im1 and Im2 expresions.
        Im1 = np.abs((X - (x1 + x2) / 2) * (x2 - x1) + (Y - (y1 + y2) / 2) * (y2 - y1)) < 0.5 * ((x2 - x1)**2 + (y2 - y1)**2)
        Im2 = np.abs((X - (x1 + x2) / 2) * (y1 - y2) + (Y - (y1 + y2) / 2) * (x2 - x1)) < 0.5 * w * np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        Im12 = Im1 * Im2

        # Npoints = Quantity points in detected rectangle
        Npoints = np.sum(np.sum(Im12))

        # Nalign = numer of aligned points in detected rectangle = k(r,Theta0)
        orient = np.angle(-(x2 - x1) + 1j * (y2 - y1))
        Nalign = np.sum(np.sum(Im12 * (np.abs(np.cos(Theta0 - orient)) < np.sin(np.pi * precis))))

        # Law k
        k_lessNalign = np.arange(0, Nalign)
        k_greatherNalign = np.arange(Nalign, Npoints + 1)

        qr = np.sum(binom.pmf(k_greatherNalign, Npoints, precis, loc = Nalign)) #loc = Nalign
        LawK = np.zeros(Npoints + 1)
        LawK[0:Nalign] = binom.pmf(k_lessNalign, Npoints, precis) / (2 * (1 - qr))
        LawK[Nalign:Npoints + 1] = binom.pmf(k_greatherNalign, Npoints, precis, loc = Nalign) / (2 * qr) ##loc = Nalign

        # Choose k pixels from the rectangle
        k = np.sum(LawK < np.random.rand(1)) - 1
        # Coord of nonzero values of Im12
        RectangleIndex = np.transpose(np.nonzero(Im12))
        # Get k random index from RectangleIndex
        k_random_index = np.random.choice(len(RectangleIndex), k, replace=False)
        # Get coordinates of k random points
        ListkR = RectangleIndex[k_random_index]

        ImR = np.mod(orient + np.pi * precis + 2 * np.pi * (1 - precis) * np.random.rand(N, M) + np.pi, 2 * np.pi) - np.pi
        ImRalig = np.mod(orient + 2 * np.pi * precis * (np.random.rand(N, M) - 0.5) + np.pi, 2 * np.pi) - np.pi
        ImR[ListkR] = ImRalig[ListkR]

        ImNew = (1 - Im12) * ImNew + Im12 * ImR

    return ImNew


#### Excercise 4 ####

def reconstruction( vectorField, amplitude, ImRect):
    N, M = vectorField.shape
    if (amplitude == 1):
        NNew = np.ones((N, M))  # uniform norm
    elif (amplitude == 100):
        NNew = np.ones((N, M)) + 100 * np.ones((N, M)) * (ImRect > 0) # uniform norm on background and R=100 in rectangles
    elif (amplitude == -1):
        NNew= np.random.rand(N, M)  #random norm

    Gu2xNew = np.zeros((2 * N, 2 * M))
    Gu2xNew[:N, :M] = NNew * np.cos(vectorField)
    Gu2xNew[N:2 * N, :M] = np.flipud(NNew * np.cos(vectorField))
    Gu2xNew[:N, M:2 * M] = np.fliplr(-NNew * np.cos(vectorField))
    Gu2xNew[N:2 * N, M:2 * M] = np.flipud(np.fliplr(-NNew * np.cos(vectorField)))

    [N2, M2] = Gu2xNew.shape

    Gu2yNew = np.zeros((2 * N, 2 * M))
    Gu2yNew[:N, :M] = NNew * np.sin(vectorField)
    Gu2yNew[N:2 * N, :M] = np.flipud(-NNew * np.sin(vectorField))
    Gu2yNew[:N, M:2 * M] = np.fliplr(NNew * np.sin(vectorField))
    Gu2yNew[N:2 * N, M:2 * M] = np.flipud(np.fliplr(-NNew * np.sin(vectorField)))

    x2 = np.zeros(N2)
    x2[:N2 // 2] = np.arange(N2 // 2)
    x2[N2 // 2 + 1:N2] = np.arange(-N2 // 2, -1)
    X2 = x2[:, np.newaxis] * np.ones((1, M2))

    y2 = np.zeros(M2)
    y2[:M2 // 2] = np.arange(M2 // 2)
    y2[M2 // 2 + 1:M2] = np.arange(-M2 // 2, -1)
    Y2 = np.ones((N2, 1)) * y2[np.newaxis, :]

    ureconstr = -np.fft.ifft2((2j * np.pi * X2 * np.fft.fft2(Gu2xNew) / N2 + 2j * np.pi * Y2 * np.fft.fft2(Gu2yNew) / M2) / (4 * np.pi**2 * ((X2 + 0.001)**2 / N2**2 + (Y2 + 0.001)**2 / M2**2)))
    Reconst = np.real(ureconstr[:N, :M])

    return Reconst

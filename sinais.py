import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
tot_bpm = []


# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid


def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


# Webcam Parameters
webcam = None
if len(sys.argv) == 2:
    webcam = cv2.VideoCapture(sys.argv[1])
else:
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

realWidth = 640
realHeight = 480
videoWidth = 320
videoHeight = 240
videoChannels = 3
videoFrameRate = 14
# videoFrameRate = 15
webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Output Videos
if len(sys.argv) != 2:
    originalVideoFilename = "original.avi"
    originalVideoWriter = cv2.VideoWriter()
    originalVideoWriter.open(originalVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate,
                             (realWidth, realHeight), True)

outputVideoFilename = "output.avi"
outputVideoWriter = cv2.VideoWriter()
outputVideoWriter.open(outputVideoFilename, cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), videoFrameRate,
                       (realWidth, realHeight), True)

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth // 2 + 5, 30)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros(bufferSize)

# Bandpass Filter for Specified Frequencies
frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros(bpmBufferSize)

i = 0
frame_no = 0

while True:

    ret, frame = webcam.read()
    if not ret:
        break

    if len(sys.argv) != 2:
        originalFrame = frame.copy()
        originalVideoWriter.write(originalFrame)

    vH = int(videoHeight / 2)
    rH = int(realHeight - videoHeight / 2)
    vW = int(videoWidth / 2)
    rW = int(realWidth - videoWidth / 2)

    detectionFrame = frame[vH:rH, vW:rW, :]

    # Construct Gaussian Pyramid
    videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
    fourierTransform = np.fft.fft(videoGauss, axis=0)

    # Bandpass Filter
    fourierTransform[mask == False] = 0

    # Grab a Pulse
    if bufferIndex % bpmCalculationFrequency == 0:
        i = i + 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
        hz = frequencies[np.argmax(fourierTransformAvg)]
        bpm = 60.0 * hz
        bpmBuffer[bpmBufferIndex] = bpm
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

    # Amplify
    filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    ifft = filtered
    filtered = filtered * alpha

    # Reconstruct Resulting Frame
    filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    outputFrame = detectionFrame + filteredFrame
    outputFrame = cv2.convertScaleAbs(outputFrame)

    bufferIndex = (bufferIndex + 1) % bufferSize

    frame[vH:rH, vW:rW, :] = outputFrame

    cv2.rectangle(frame, (vW, vH), (rW - vW, rH - vH),
                  boxColor, boxWeight)
    if i > bpmBufferSize:
        cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
        tot_bpm.append(bpmBuffer.mean())
        outputVideoWriter.write(frame)
    else:
        cv2.putText(frame, "Calculating BPM...", loadingTextLocation, font, fontScale, fontColor, lineType)

    if len(sys.argv) != 2:
        cv2.imshow("Webcam Heart Rate Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # teste___________________________________
            # tot_time = []
            # cap = cv2.VideoCapture(r'D:\dev\trabalho_final_sinais\output.avi')
            #
            # frame_no = 0
            # while cap.isOpened():
            #     frame_exists, curr_frame = cap.read()
            #     if frame_exists:
            #         print("for frame : " + str(frame_no) + "   timestamp is: ",
            #               str(round(cap.get(cv2.CAP_PROP_POS_MSEC), 2)))
            #     else:
            #         break
            #     frame_no += 1
            #     tot_time.append(round(cap.get(cv2.CAP_PROP_POS_MSEC), 2))
            #     print(max(tot_time))
            # cap.release()
            #
            # time_step = 66.67
            # time_vec = np.arange(min(tot_time), max(tot_time), time_step)
            #
            # if len(time_vec) > len(tot_bpm):
            #     aux = len(time_vec) - len(tot_bpm)
            #     for i in range(aux):
            #         last, time_vec = time_vec[-1], time_vec[:-1]
            #
            # if len(time_vec) < len(tot_bpm):
            #     aux = len(tot_bpm) - len(time_vec)
            #     for i in range(aux):
            #         tot_bpm.pop()
            #
            # plt.figure(figsize=(6, 5))
            # plt.plot(time_vec, tot_bpm, label='Original signal')
            # plt.show()
            #
            # # teste__________________________________
            #
            # real = np.array([i[0] for i in ifft.real])
            # imag = np.array([i[0] for i in ifft.imag])
            #
            # real = np.array([i[0] for i in real])
            # imag = np.array([i[0] for i in imag])
            #
            # real = np.array([i[0] for i in real])
            # imag = np.array([i[0] for i in imag])
            #
            # plt.plot(frequencies, real, label='real')
            # plt.plot(frequencies, imag, '--', label='imaginary')
            # plt.legend()
            # plt.show()
            # print(time_vec)
            # print(tot_bpm)
            #  teste________________________________
            break

webcam.release()
# cv2.destroyAllWindows()
outputVideoWriter.release()
if len(sys.argv) != 2:
    originalVideoWriter.release()

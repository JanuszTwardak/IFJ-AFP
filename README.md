# AFP events anomaly detection using simple convolutional autoencoder

## INTRODUCTION

### Preface

This project has been made during my 1-month internship in Institute of Nuclear Physics of the Polish Academmy of Sciences.
It was mainly done to deepen and make use of my knowledge about machine learning and neural networks thus decision to represent hits as planes and to use autoencoders. In reality it isn't optimal solution because of some problems:
* images representation is really sparse and it cannot include too much additional information about events,
* we can't predefine what anomaly event is - we want determine it without human's bias. This results with:
  * learning dataset containing anomalies, which conflicts with the whole idea of using autoencoders to anomaly detection,
  * we can't make automatic model validation, as it is needed to check every classified event by qualified physicist who will analyze results

All of the above points make practical use of this code highly ineffective thus my decision to quit further development. Though I'm satisfied with internship and process of creating code as I have learned a lot, both about high energy physics and deep machine learning.

After internship I have been continuing this research project. In my other repository I am working on tackling this problem from different perspective by using Isolation Forest algorithm. It's aiming strictly to produce proper results, which differs it from this project.

### About AFP experiment (mild physics included)

The ATLAS Forward Proton (AFP) project promises a significant extension to the physics reach of ATLAS by tagging and measuring the momentum and emission angle of very forward protons. This enables the observation and measurement of a range of processes where one or both protons remain intact which otherwise would be difficult or impossible to study. Such processes are typically associated with elastic and diffractive scattering, where the proton radiates a virtual colorless “object,” the so-called Pomeron, which is often thought of as a non-perturbative collection of soft gluons.

<img width="700" alt="3" src="https://user-images.githubusercontent.com/50775382/160122233-729fd02d-825b-43ea-ac63-8c4d37e00267.jpg">

Diffractive protons are usually scattered at very small angles (hundreds of micro radians. In order to measure them, special devices that allow for the detector movement (so-called Roman pots, RP) are commonly used. The ATLAS Forward Proton (AFP) are placed symmetrically with respect to the ATLAS IP at 204 m and 217 m. Stations located closer to the IP contain the tracking detectors, whereas the further ones are equipped with tracking and timing devices. The reconstruction resolution of tracking detectors is estimated to be of 10 and 30 μm in x and y, respectively. The precision of Time-of-Flight measurement is expected to be of about 20 ps.


<img width="400" alt="3" src="https://user-images.githubusercontent.com/50775382/160123664-fd4a3ac3-0732-4032-8f72-ea9e0e7b93b4.jpg">


Readings from detectors can be very different depending which event type has occured. It can either show one, clear diffractive proton hit, a lot of different hits (shower), missing hits due to detector fault or some other strange incidents (examples show below). Our intend was to filter these events to receive dataset that contains only single proton hits. This will allow to conduct further researches on this phenomenon as well as it might find some events we would wrongly classify as anomalies, but in reality they are truly diffractive protons hits.

Examples of different, independent registered events (projected from 2D to 1D plane due to better readability):


<img width="300" alt="3" src="https://user-images.githubusercontent.com/50775382/160122746-f760fff9-6372-4a52-b15c-5bd4e7ae8b2c.png"> <img width="244" alt="5" src="https://user-images.githubusercontent.com/50775382/160122789-e3e609d3-90a8-4b9c-876d-a5f5e0e67508.png"> <img width="230" alt="4" src="https://user-images.githubusercontent.com/50775382/160122801-135382cf-7f68-4818-9a88-5704c2e7540d.png">


## Implementation
Code was written to be run on Google Colab. Preprocessed data in form of NumPy Array needs to be loaded from Google Drive (you need to upload them earlier). 

Notebook also has cells to load already trained model


## Results

Each image below represents one event from single detector. Each image consists of two rows of images, each row contains four planes.
First row represents original data from detector.
Second row represents images compressed and reconstructed by our autoencoder. 
After reconstruction it calculates difference from the original image (mean square error) and then compares it against threshold calculated statistically. If error is bigger than threshold, event is classified as anomaly.



### Normal events:

<img width="500" alt="3" src="https://user-images.githubusercontent.com/50775382/160127695-88dec52d-5118-452a-bfb2-d400ee2f4f06.png"> 

#### - - -
<img width="500" alt="3" src="https://user-images.githubusercontent.com/50775382/160127738-3d5c8106-8a83-47eb-95ef-440875ff6deb.png"> 



### Anomalies:

<img width="500" alt="3" src="https://user-images.githubusercontent.com/50775382/160127058-462c0660-c440-49c9-b70a-7c4d6cc0ba14.png"> 

#### - - -

<img width="500" alt="3" src="https://user-images.githubusercontent.com/50775382/160126633-0423f8f7-0bb7-484f-9b7c-68035c798cba.png"> 

#### - - -

<img width="500" alt="3" src="https://user-images.githubusercontent.com/50775382/160127004-dc427eec-de2d-471b-9945-da28c1aabbaa.png"> 

#### - - -

<img width="500" alt="3" src="https://user-images.githubusercontent.com/50775382/160127013-f7729399-0458-4311-95ac-3a4c415c1b6f.png"> 

#### - - -

<img width="500" alt="3" src="https://user-images.githubusercontent.com/50775382/160127031-5fcf81c3-8e11-449c-a272-78ff966d4556.png"> 







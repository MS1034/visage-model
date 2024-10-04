# Face-Aware Image Segregation for Personal Albums

- 1 Introduction Contents
  - 1.1 Background
  - 1.2 Problem Statement
  - 1.3 Proposed Solution
- 2 Literature Review
  - 2.1 Face Detection and Recognition Using Siamese Neural Networks
  - 2.2 Face Recognition based on Siamese Convolutional Neural Network using Kivy
  - 2.3 FaceNet: A Unified Embedding for Face Recognition and Clustering
- 3 Methodology & System Architecture
  - 3.1 Image Datasets
  - 3.2 Data Augmentation & Pre Processing
  - 3.3 Models Architecture
    - 3.3.1 SNN
    - 3.3.2 FaceNet
  - 3.4 Loss Functions
  - 3.5 Pipeline
    - 3.5.1 Input
    - 3.5.2 Face Detection
    - 3.5.3 Face Recognition
    - 3.5.4 Image Segregation
    - 3.5.5 Output
  - 3.6 Training Testing Model
- 4 Experimental Results
  - 4.1 Face Detection
  - 4.2 Face Recognition
  - 4.3 Scenario I: Friends Album & Web Cam Subject
  - 4.4 Scenario II: Friends Album & Face Aware Image Segregation
  - 4.5 Scenario III: Fam2a (The Images of Group Dataset) and Image Segregation
- 5 Visage: Real World Implementation
- 6 Challenges
- 7 Future Work
- 8 Conclusion


## 1 Introduction

### 1.1 Background

Digital photography became common with the integration of cameras with mobile phones. In
this era of digital photography, large collections of images are captured on different occasions.
Organizing these photos is a challenge, especially for professional photographers. There is a
dire need to segregate images based on individuals present in images. Many traditional photo
management tools lack this feature.

### 1.2 Problem Statement

There is a need for an intelligent tool that automates the process of segregating images in an
album based on the faces of individuals present in photos.

### 1.3 Proposed Solution

The objective of this project is to develop a Face-Aware Image Segregation tool that can
automatically categorize images in an album based on the faces of individuals present in photos.
The tool will use computer vision and image processing techniques to detect and recognize faces
in images.

## 2 Literature Review

### 2.1 Face Detection and Recognition Using Siamese Neural Networks

**Explanation**:

The paper presents a novel approach to face detection and recognition using a Siamese neural
network. It addresses the growing importance of biometric identification in various scenar-
ios such as airport security, access to military zones, and social service authentication. The
Siamese neural networkis designed to compare two facial images and judge their similarity,
enabling accurate face recognition. This approach utilizes deep learning techniques, specifically
convolutional neural networks (CNNs).

The methodology involves collecting a dataset of facial images from various sources, including
the computer science department at the University of Chlef and the ”labelled faces in the
wild” dataset. The collected images are preprocessed and used to train the Siamese neural
network. The training process involves optimizing the network’s parameters using techniques
likecontrastive loss functions and triplet loss. The trained model achieves high accuracy
in recognizing faces, with experimental results indicating superior performance compared to
existing methods.[1]

### 2.2 Face Recognition based on Siamese Convolutional Neural Net-

### work using Kivy

**Explanation**:

The paper presents a thorough study on recognizing human faces, which is important for many
use cases. It talks about the difficulties faced due to factors like lighting, emotion, and facial
expressions. The study focuses on creating a mobile app for recognizing faces using a special


type of neural network called Siamese CNN. This network is good at learning similarities
between two objects, which is useful for tasks where you only have one example to learn from.

They tested this method using a large dataset of 9,000 face images and found that it works well,
especially when they used augmentation techniques to make the dataset bigger. The results
showed high accuracy, with the system being able to correctly recognize faces in 98% of cases.
The method involves training the network with batches of images, where some images show
the same person and others show different people. They also used a technique called the Kivy
framework to make the app easy to use on mobile phones.

The study concludes by saying that this method of using Siamese CNN for face recognition is
very effective, especially when combined with augmentation techniques. They suggest that in
the future, they could try it with different datasets and try to make it even more accurate by
using even more data.[2]

### 2.3 FaceNet: A Unified Embedding for Face Recognition and Clus-

### tering

**Explanation:**

The paper presents a system called FaceNet, which directly learns a mapping from face images to
a compact Euclidean space where distances directly measures face similarity. FaceNet utilizes
a deep convolutional neural network (CNN) trained to directly optimize the embedding. It
does not rely on an intermediate bottleneck layer as in previous methods. The network is
trained using triplets of images: an anchor image, a positive image (of the same person as
the anchor), and a negative image (of a different person). The network aims to minimize the
distance between the anchor and the positive image while maximizing the distance between the
anchor and the negative image. This triplet loss function helps the network learn meaningful
representations that capture facial similarities. Once the embedding space is learned, tasks
like face recognition, verification (determining if two images are of the same person), and
clustering (grouping similar faces) can be performed using standard techniques with FaceNet
embeddings as feature vectors. FaceNet achieves state-of-the-art performance on benchmark
datasets, significantly reducing error rates compared to previous methods.The use of a compact
embedding space allows for efficient storage and retrieval of facial representations. The paper
introduces the concept of harmonic embeddings, describing different versions of face embeddings
that are compatible with each other, allowing for direct comparison between them.[3]

## 3 Methodology & System Architecture

### 3.1 Image Datasets

Labeled Faces in the Wild (LFW) is a dataset for labeled images. It is preprocessed to extract
face images and their corresponding labels. Each input batch will consist of three Images:

- Anchor
- Positive
- Negative

In addition to using established datasets like LFW for training and evaluation, dateset The
Images of Groups Dataset is used to evalute the implementation of model.


Figure 1: Upper partition: Images from the Group dataset treated as event images. Lower
partition: Image pairs from Labeled Faces in the Wild (LFW) dataset.

### 3.2 Data Augmentation & Pre Processing

In order to introduce modular variability, techniques such as rotation, flipping, cropping, scal-
ing, translation, and color jittering are often used. These transformations help the model
become invariant to variations in orientation, position, scale, and color, making it more dy-
namic. Although the data was already augmented and pre-processed, it is recommended to
utilizeAlbumentations andOpenCV for further enhancements. For more details, refer to the
Future Worksection.

### 3.3 Models Architecture

#### 3.3.1 SNN

The Siamese Neural Network (SNN) is a type of neural network architecture designed for
learning similarity between pairs of data. In the provided architecture, the SNN consists of
two identical subnetworks (denoted as A to R and A’ to R’), each with a similar structure
comprising convolutional layers (B, F, J, and N), batch normalization layers (C, G, K, and
O), ReLU activation functions (D, H, L, and P), and max-pooling layers (E, I, M). These
subnetworks share the same weights and are fed with pairs of input data simultaneously. The
output of each subnetwork is then concatenated (Q) and passed through a dense layer (R) with
sigmoid activation, generating a similarity score between the input pairs. This architecture
allows the network to learn feature representations that capture the similarity between the
input data pairs.

```markdown
A[Input Shape] --> B[Conv2D(64, (10, 10), name='Conv1')]
B --> C[BatchNormalization()]
C --> D[Activation("relu")]
D --> E[MaxPooling2D()]
E --> F[Conv2D(128, (7, 7) name='Conv2')]
F --> G[BatchNormalization()]
G --> H[Activation("relu")]
H --> I[MaxPooling2D()]
I --> J[Conv2D(128, (4, 4) name='Conv3')]
J --> K[BatchNormalization()]
K --> L[Activation("relu")]
L --> M[MaxPooling2D()]
M --> N[Conv2D(256, (4, 4) name='Conv4')]
N --> O[BatchNormalization()]
O --> P[Activation("relu")]
P --> Q[Flatten()] --> R[Dense(4096, activation='sigmoid')]
```

#### 3.3.2 FaceNet

FaceNet is a deep learning system designed for face recognition tasks, particularly face verifi-
cation and clustering. The architecture consists of a series of convolutional layers (C, H, L),
followed by batch normalization (D, I, M) and ReLU activation functions (E, J, N). These
layers are structured to extract meaningful features from input face images. Additionally, the
architecture includes inception blocks (Q, T, V) to capture more complex patterns and improve
representation learning. The output of the network is a 128-dimensional embedding (Z) that
represents the face image in a compact Euclidean space, where distances directly correspond to
measures of face similarity. This embedding is then normalized (AA) and used for tasks such
as face recognition and clustering. Overall, FaceNet’s architecture enables it to efficiently learn
discriminative feature representations for accurate and scalable face recognition.

```markdown
A[Input(input_shape)] --> B[ZeroPadding2D((3, 3))]
B --> C[Conv2D(64, (7, 7), strides=(2, 2), name='conv1')]
C --> D[BatchNormalization(axis=1, name='bn1')]
D --> E[Activation('relu')]
E --> F[ZeroPadding2D((1, 1))]
F --> G[MaxPooling2D((3, 3), strides=2)]
G --> H[Conv2D(64, (1, 1), strides=(1, 1), name='conv2')]
H --> I[BatchNormalization(axis=1, epsilon=0.00001, name='bn2')]
I --> J[Activation('relu')]
J --> K[ZeroPadding2D((1, 1))]
K --> L[Conv2D(192, (3, 3), strides=(1, 1), name='conv3')]
L --> M[BatchNormalization(axis=1, epsilon=0.00001, name='bn3')]
M --> N[Activation('relu')]
N --> O[ZeroPadding2D((1, 1))]
O --> P[MaxPooling2D(pool_size=3, strides=2)]
P --> Q[inception_block_1a]
Q --> R[inception_block_1b]
R --> S[inception_block_1c]
S --> T[inception_block_2a]
T --> U[inception_block_2b]
U --> V[inception_block_3a]
V --> W[inception_block_3b]
W --> X[AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
data_format='channels_first')]
X --> Y[Flatten()] --> Z[Dense(128, name='dense_layer')]
Z --> AA[Lambda(lambda x: K.l2_normalize(x, axis=1))]
AA --> AB[Model(inputs=X_input, outputs=X, name='FaceRecoModel')]
```


### 3.4 Loss Functions

The choice of loss function is crucial in training for face-aware image segregation. Commonly
used loss functions include contrastive loss and triplet loss. Contrastive loss penalizes the model
when the similarity between images from the same individual is less than a certain margin, while
triplet loss enforces that the distance between images from the same individual is smaller than
the distance between images from different individuals by at least a margin. I have usedtriplet
loss.

triplet(A,P,N) = max(d(A,P)−d(A,N) +α,0)


### 3.5 Pipeline

#### 3.5.1 Input

The input data is provided in a folder namedEvent Images, containing all the images that
need to be sorted.

Figure 3: Image captured from the webcam for comparison with the album. This input image
serves as the basis for model to compare with the existing album.

#### 3.5.2 Face Detection

Face detection is performed usingOpenCV. While the detection may occasionally fail, the overall
accuracy remains satisfactory, achieving approximately 95.3%. The output of the face detection
stage is a list of all detected faces, extracted using theextractfacesfunction. Each face is
accompanied by its aligned coordinates, including details such asx,y, width, height, and the
positions of the left and right eyes.

#### 3.5.3 Face Recognition

FaceNetis employed for face recognition tasks. Theverifyfunction ofFaceNetis utilized to
compare pairs of face images, returning a boolean value indicating whether the faces belong to
the same person. A similarity threshold, set as a hyperparameter, governs this process, with a
typical values of 0. 6 , 0. 7 , 0. 75 , 0 .8.

#### 3.5.4 Image Segregation

After detecting and aligning faces from the input images, a set of unique faces is created by
removing similar ones. Each unique face is stored in a separate folder within the Output
directory, named according to the individual it represents. Additionally, all images containing
the same face are grouped together in their respective folders. This segregation process ensures
that each folder within theOutputdirectory contains images featuring only a single individual.


Figure 4: Annotated image showcasing facial landmarks. The markings depict key points such
as the eyes and facial contours, aiding in facial recognition algorithms.

#### 3.5.5 Output

The output of the pipeline consists of folders containing images segregated based on the faces
detected. Each folder is labeled with the identifier of the corresponding face, and it contains all
the images where that particular face appears. Additionally, a folder namedFacesis created
within theOutputdirectory, containing individual images of each unique face identified during
the process.

```markdown
- Project
- model
  - facenet
    - [facenet_model.py](model/facenet/utils/inception_blocks_v2.py)
  - SNN
    - [snn_model_code.py](model/SNN/SNN.ipynb)
- input
  - Event-Images
    - [image1.jpg](input/Event-Images/image1.jpg)
    - [image2.jpg](input/Event-Images/image2.jpg)
    - ...
- output
  - Faces
    - Face
      - [imag{Pipeline}e1.jpg](input/Event-Images/image1.jpg)
        - [image2.jpg](input/Event-Images/image2.jpg)
      - ...
    - Face
      - [image5.jpg](input/Event-Images/image1.jpg)


- [image2.jpg](input/Event-Images/image2.jpg)
  - ...
- FaceN
- [image7.jpg](input/Event-Images/image1.jpg)
- [image8.jpg](input/Event-Images/image2.jpg)
- ...
- [main.py](main.py)
```

### 3.6 Training Testing Model

The model was trained using a powerful online platform (Colab) and trained on a large collection
of real-world face images called the Labeled Faces in the Wild (LFW) dataset. This dataset
includes a wide variety of faces from different people. The transfer learning techniques were
used.

To ensure the model’s accuracy, the dataset was split into two parts: a training set and a testing
set. The training set was used to teach the model by showing it many labeled images. By looking
at these examples, the model learned to identify patterns and features that distinguish different
faces. After training, the model’s performance was evaluated using the separate testing set.
This testing showed how well the model could recognize faces from images it had never seen
before. In short, this process helped us understand how good the model is at recognizing faces
in general. The results are given in the figure below.

```
Figure 5: Figure showing result of tarining testing Siamese Neural Network.
```

## 4 Experimental Results

### 4.1 Face Detection

Total Images (Batch Size): 754
Actual Faces: 1951
Accuracy: 0.
Precision: 0.
Recall: 0.
F1-Score: 0.

### 4.2 Face Recognition

The evaluation results for three different scenarios are presented below:

### 4.3 Scenario I: Friends Album & Web Cam Subject

In Scenario I, the evaluation was conducted using images from the ”friends” Album, compare
with the subject, Subhan.

```javascript
{'True Positive': 28, 'False Positive': 22, 'True Negative': 0, 'False Negative': 0, 'Accuracy': 0.56, 'Precision': 0.56, 'Recall': 1.0, 'F1 Score': 0.717948717948718}
```



### 4.4 Scenario II: Friends Album & Face Aware Image Segregation

Scenario II involved assessing the ”friends” album by segregates images based on detected faces.

```javascript
{'True Positive': 78, 'False Positive': 23, 'True Negative': 0, 'False Negative': 3, 'Accuracy': 0.75, 'Precision': 0.7722772277227723, 'Recall': 0.9629629629629629, 'F1 Score': 0.8571428571428571}
```



### 4.5 Scenario III: Fam2a (The Images of Group Dataset) and Image Segregation

Scenario II involved assessing the ”Fam2a” album by segregates images based on detected faces.

```javascript
{'True Positive': 85, 'False Positive': 4, 'True Negative': 6, 'False Negative': 14, 'Accuracy': 0.8348623853211009, 'Precision': 0.9550561797752809, 'Recall': 0.8585858585858586, 'F1 Score': 0.9042553191489363}
```



## 5 Visage: Real World Implementation

Introduction to Visage

Visage is a real-world implementation of our project, deployed on Azure and built using Django
and microservices architecture. It offers effortless sharing of event photos with AI-powered face


```
Figure 6: Visage: Real World Implementation
```

recognition, eliminating the need for manual sorting and ensuring unforgettable memories are
easily organized.

Key Features will be:

- Barcode Sharing: Simplify photo distribution with barcode technology.
- Access Attributes:Customize access settings for sharing memories securely.
- No Quality Loss:Preserve image quality throughout the photo journey.
- Face Recognition: Automatically organize memories using facial recognition technol-
  ogy.
- Data Security:Ensure data protection with robust encryption measures.
- Google Drive Plugin: Seamlessly integrate Visage with Google Drive for enhanced
  storage capabilities.

## 6 Challenges

The were the chanllages faced during the project:

- False Positive Faces: One of the main challenges was the occurrence of false positive
  detections, where the system incorrectly identified non-facial objects or patterns as faces.
  This led to inaccuracies in the recognition results and decreased the overall reliability of
  the system.
- Scalability and Efficiency for Large Datasets: As the size of the dataset and the
  number of faces to be recognized increased, the system faced challenges in terms of scala-
  bility and computational efficiency. Optimizing the model and its deployment for larger-
  scale applications was a key concern.
- Similar-looking Faces and False Recognition: Another significant challenge was
  distinguishing between faces that shared similar features or appearances. In some cases,the system mistakenly recognized two different individuals as the same person, leading to
  false recognition errors.

## 7 Future Work

In future work, we plan to explore the following areas:

- Further investigate the impact of data augmentation techniques, such as Albumentations,
  on model performance.
- Experiment with different face detection algorithms to improve the face detection specially
  false positives in high resolution images.
- Optimize the efficiency for embeded Systems.
- Explore options to scale the model for larger datasets and its deployment on distributed
  systems.
- Development of real-world project in one of the python web framework (django, flask,
  FastAPI).

## 8 Conclusion

In a nustshell, the Face-Aware Image Segregation tool is a significant advancement in orga-
nizing large photo collections based on faces. It uses modern CV techniques to automatically
sort images by faces. This tool benefits both professional photographers and everyday users by
streamlining photo management. The tool’s success in detecting and recognizing faces relies
on models like Siamese Neural Networks and FaceNet. While there are challenges like occa-
sional misidentifications and limitations on how many photos it can handle, the project has
established a strong foundation for future improvements. Further research on data enhance-
ment, optimization for smaller devices, and using distributed systems can improve the tool’s
performance and scalability. Overall, the Face-Aware Image Segregation tool has the potential
to significantly change how we organize and manage photos, offering a user-friendly way to
preserve memories.

## References

[1] Nesrine Djawida Hamdani, Nassima Bousahba, Ahmed Houdaifa Bousbai, and Amina
Braikia. Face detection and recognition using siamese neural network.International Journal
of Computing and Digital Systems, 2023.

[2] Yazid Aufar and Imas Sukaesih Sitanggang. Face recognition based on siamese convolutional
neural network using kivy framework. Indonesian Journal of Electrical Engineering and
Computer Science, 2022.

[3] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding
for face recognition and clustering.arXiv preprint arXiv:1503.03832, 2015.

## Flow Decomposition
1. Initially created Flow decomposition with original architecture based on the feature extractor and 
flow estimator, but the cost volume output from the feature extractor and result from the flow estimator would
need to be trained with the output of a pre-trained model, so we will be using a newer and faster model to output 
the optical flow. 
2. I experimented with algorithmic approach using pyflow and cv2, but finding the flow movements are picked up
on the movement of the background, so a pre-trained network will be used instead. pictures in files/opticalFlow.
3. spynet will be used instead of PCW net, as it is light weight and only uses one network

## Dataset
- Downloaded and will be using De-fencing dataset for testing the pretrained. Sample image in 
files/deFencingSamples
- Alignment is in FlowDecomposer and aligns based on a keyframe using homography transformation.
The flow, and frames will use the alignment to align the frames before concatenating as an input to the background 
network.
- (Not done) To generate the synthetic dataset, a homography transform will be used on one image,
creating 4 other transformated images to imitate movement, then an occlusion image will be synethesized
on top in one place to generate new datasets.
- (Not done) Alpha map will be extracted from images
- (Not done) Difference map will be extracted

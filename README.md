# SAMv2-YOLO11

## Autodistill for auto annotation

AutoDistill introduces a revolutionary approach to dataset labeling and model training by automating the process of dataset annotation. Unlike traditional methods that rely on manual human intervention, AutoDistill leverages advanced techniques like GroundingDINO and SAM to automatically label datasets. This groundbreaking framework allows developers to train small and fast supervised models by using big and slow foundation models (i.e., models trained on broad general datasets that are applicable to a wide range of tasks), without the need for extensive human effort.

To use AutoDistill, developers input unlabeled data into a Base Model, which employs an Ontology to label the dataset. The labeled dataset is then utilized to train a Target Model, which outputs a finely tuned Distilled Model tailored to perform a specific task. This streamlined process eliminates the burden of manual data annotation, enabling developers to expedite the model development cycle and deploy custom models at the edge with unparalleled efficiency.
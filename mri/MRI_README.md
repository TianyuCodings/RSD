## RSD: MRI experiments


### Download pre-trained models

EDM (teacher) models: The checkpoints are available [here](https://utexas.box.com/s/axofnwib9kukdpa92ge4ays87dmuvpf7). To download from the terminal, simply run:

```
wget -v -O ambient_models.zip -L https://utexas.box.com/shared/static/axofnwib9kukdpa92ge4ays87dmuvpf7.zip
```

### Download dataset

**Option 1:** you can directly download the [processed data](https://huggingface.co/datasets/yasiz/ambient_mri_data)

**Option 2:**

For the experiments, we used a pre-processed version of NYU's [FastMRI dataset](https://fastmri.med.nyu.edu/). 

Reference dataset processing script:  [process_data.py](./process_data.py)

 

## Training New Models

stage 1: pretraining is done by the EDM checkpoint

stage 2: run 
```
bash ./sid_train.sh
```

checkout the `sid_train.sh` bash script for different settings of `R`.

Evaluation is done during distillation.
# STICM: Spatiotemporal information conversion machine for time-series forecasting
This respository includes codes and datasets for paper "Spatiotemporal information conversion machine for time-series forecasting".

 The [paper](https://doi.org/10.1016/j.fmre.2022.12.009) is online on [Fundamental Research](https://www.sciencedirect.com/journal/fundamental-research) journal. 

## Spatiotemporal information conversion machine for time-series forecasting
- A spatiotemporal information conversion machine (STICM), was developed to efficiently and accurately render a multistep-ahead prediction of a time series by employing a spatial-temporal information (STI) transformation.

## Data avalability

- We put data file including the locations of traffic loops in folder [datasets/traffic](./datasets/traffic).
- The locations of all 155 meteorological stations used in wind speed dataset are provided in folder [datasets/ws](./datasets/ws).
- Other dataset are uploaded to Google Drive, and can be downloaded [here](https://drive.google.com/file/d/1THvn_D5TG_cW5rHVTjxOFQN7i0mu1Dgj/view?usp=sharing).

## Environment requirements

- python = 3.6
- tenforflow = 2.1
- cuda-version = 10.1
- cudnn-version = 7.6.5

## Training and making predicitons

- We release the sample training codes and predicting codes corresponding to the Lorenz dataset, which is located at folder `experiment/multi_sample_symmetric/`. The script `train.py` is used for training and the script `eval.py` is used for evaluation after training the model. 

- We can make predictions on other datasets by modify the given sample codes for Lorenz dataset.


## Examples

- The example movie in terms of traffic speed prediction mentioned in our paper is given in folder [example_movies/traffic_movie.mp4](./example_movies/traffic_movie.mp4).

## Citation

If you find this repository useful in your research, please consider citing the following papers:
```
@article{peng2022spatiotemporal,
  title={Spatiotemporal information conversion machine for time-series forecasting},
  author={Peng, Hao and Chen, Pei and Liu, Rui and Chen, Luonan},
  journal={Fundamental Research},
  year={2022},
  publisher={Elsevier}
}
```
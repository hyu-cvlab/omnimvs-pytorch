# OmniMVS
This repository contains python codes for paper, "End-to-End Learning for Omnidirectional Stereo Matching with Uncertainty Prior" (TPAMI).

**Contact**: Changhee Won (changhee.1.won@gmail.com)


## Prerequisites
### List of code/library dependencies
- Pytorch (tested on 1.5.1)
- ``` pip install numpy scipy matplotlib pyyaml EasyDict scikit-image ```

## How to run
### Test (run_test_omnimvs.py)
- Pretrained weights: 
    - [[omnimvs_plus_ft.pt](https://bit.ly/3nlRrj4)]
    - [[tiny_plus_ft.pt](https://bit.ly/2ERUQob)]
- Set **arguments** in the script
- Run ``` python run_test_omnimvs.py [path_to_pt_file] [dbname]```

### Example results
<img src="https://user-images.githubusercontent.com/7540390/94922956-12466000-04f6-11eb-9944-a02384d68cb3.png" width=42.7%><img src="https://user-images.githubusercontent.com/7540390/94922979-170b1400-04f6-11eb-9b6c-85caac809d5f.png" width=50%>

## Dataset
You can download the synthetic datasets in the [project page](http://cvlab.hanyang.ac.kr/project/omnistereo).

The directory structure should be like this:
```
[db_root]/[dbname]/[cam%d]/[%05d.png]
                  /omnidepth_gt_640/%05d.tiff  # not necessary
                  /config.yaml
                   ...
(e.g.)
data/sunny/
          /cam1/
          /cam2/
          /cam3/
          /cam4/
          /omnidepth_gt_640/
          /config.yaml
```

## Citation & Acknowledgement
We founded a start-up company [MultiplEYE co. ltd.](http://multipleye.co) based on this research.
```
@article{won2020end,
    title={End-to-End Learning for Omnidirectional Stereo Matching with Uncertainty Prior},
    author={Won, Changhee and Ryu, Jongbin and Lim, Jongwoo},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)},
    year={2020},
}
```


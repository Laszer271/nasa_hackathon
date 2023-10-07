---
license: apache-2.0
tags:
- Pytorch
- Geospatial
- Temporal ViT
- Vit
---

### Model and Inputs
Prithvi is a first-of-its-kind temporal Vision transformer pre-trained by the IBM and NASA team on contiguous US Harmonised Landsat Sentinel 2 (HLS) data. The model adopts a self-supervised encoder developed with a ViT architecture and Masked AutoEncoder (MAE) learning strategy, with an MSE loss function. The model includes spatial attention across multiple patches and also temporal attention for each patch.

![](GFM.png)

The model accepts remote sensing data in a video format (B, C, T, H, W). Note that the temporal dimension (T) is very important in this application and not present in most other works around remote sensing modeling. The ability to handle a time series of remote sensing images can benefit a variety of downstream tasks (e.g. Burn Scars segmentation, Flood Segmentation, Land Cover Classification). The model can also handle static imagery which can be fed into the model with T=1.

### Pre-training
The model was pre-trained with NASA's HLS V2 L30 product (30m granularity) from the contiguous United States. The bands that were used are the following:

1.  Blue
2.  Green
3.  Red
4.  Narrow NIR
5.  SWIR 1
6.  SWIR 2

### Code
The model follows the [original MAE repo](https://github.com/facebookresearch/mae) with some modifications including:

1. replace 2D patch embed with 3D patch embed;
2. replace 2D positional embed with 3D positional embed;
3. replace 2D patchify and unpatchify with 3D.
4. adding infrared bands besides RGB

### Inference and demo
There is an inference script (`Prithvi_run_inference.py`) that allows to run the image reconstruction on a set of HLS images assumed to be from the same location at different time steps(see example below). These should be provided in chronological order in geotiff format, including the channels described above (Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2) in reflectance units. There is also a **demo** that leverages the same code [here](https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-demo).

```
python Prithvi_run_inference.py --data_files t1.tif t2.tif t3.tif --yaml_file_path /path/to/yaml/Prithvi_100.yaml --checkpoint /path/to/checkpoint/Prithvi_100.pth --output_dir /path/to/out/dir/ --mask_ratio 0.5
```

### Finetuning examples
Examples of finetuning the model for image segmentation using the mmsegmentation library are available through Hugging Face (e.g. [burn scars segmentation](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-burn-scar), [flood mapping](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-sen1floods11), and [multi temporal crop classification](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification)), with the code used for the experiments available on [github](https://github.com/NASA-IMPACT/hls-foundation-os/tree/main/fine-tuning-examples). This also contains instructions to finetune the model for flood detection on the popular open access [sen1floods11 dataset](https://github.com/cloudtostreet/Sen1Floods11).

### Feedback

Your feedback is invaluable to us. If you have any feedback about the model, please feel free to share it with us. You can do this by submitting issues on our open-source repository, [hls-foundation-os](https://github.com/NASA-IMPACT/hls-foundation-os/issues), on GitHub.

### Citation

If this model helped your research, please cite `Prithvi-100M` in your publications. Here is an example BibTeX entry:

```
@misc{Prithvi-100M,
    author          = {Jakubik, Johannes and Chu, Linsong and Fraccaro, Paolo and Gomes, Carlos and Nyirjesy, Gabby and Bangalore, Ranjini and Lambhate, Devyani and Das, Kamal and Oliveira Borges, Dario and Kimura, Daiki and Simumba, Naomi and Szwarcman, Daniela and Muszynski, Michal and Weldemariam, Kommy and Zadrozny, Bianca and Ganti, Raghu and Costa, Carlos and Edwards, Blair & Watson, Campbell and Mukkavilli, Karthik and Schmude, Johannes & Hamann, Hendrik and Robert, Parkin and Roy, Sujit and Phillips, Christopher and Ankur, Kumar and Ramasubramanian, Muthukumaran and Gurung, Iksha and Leong, Wei Ji and Avery, Ryan and Ramachandran, Rahul and Maskey, Manil and Olofossen, Pontus and Fancher, Elizabeth and Lee, Tsengdar and Murphy, Kevin and Duffy, Dan and Little, Mike and Alemohammad, Hamed and Cecil, Michael and Li, Steve and Khallaghi, Sam and Godwin, Denys and Ahmadi, Maryam and Kordi, Fatemeh and Saux, Bertrand and Pastick, Neal and Doucette, Peter and Fleckenstein, Rylie and Luanga, Dalton and Corvin, Alex and Granger, Erwan},
    doi             = {10.57967/hf/0952},
    month           = aug,
    title           = {{Prithvi-100M}},
    repository-code = {https://github.com/NASA-IMPACT/hls-foundation-os},
    year            = {2023}
}
```

This is the readme text for the preparation of dataset.

part1: Retinal images
Please download corresponding dataset and preproce them as followed:

*1.1 Large raw retinal images
1.1.1 Kaggle Diabetic Retinopathy Detection
. Official website: https://www.kaggle.com/c/diabetic-retinopathy-detection/overview.
. According to the competition host. This dataset can be used for research purpose and following article shoould be cited: 
  Cuadros J, Bresnick G. EyePACS: An Adaptable Telemedicine System for Diabetic Retinopathy Screening. Journal of diabetes science and technology (Online). 2009;3(3):509-516.
. Unzip the training dataset into ./retinal_images/original_data/kaggle_training.


*1.2 Retinal vessel segmentation dataset
1.2.1 RITE dataset
. Official website: https://medicine.uiowa.edu/eye/rite-dataset.
. Plaese cite: 
  Hu Q, Abràmoff MD, Garvin MK. Automated separation of binary overlapping trees in low-contrast color retinal images. Med Image Comput Comput Assist Interv. 2013;16(Pt 2):436-43. PubMed PMID: 24579170
. Unzip the data into ./retinal_images/original_data/RITE. 
1.2.2 STARE dataset
. Official website: http://cecas.clemson.edu/~ahoover/stare/
. Unzip the dataset into ./retinal_images/original_data/STARE
1.2.3 CHASEDB dataset:
. Official website: https://blogs.kingston.ac.uk/retinal/chasedb1/
. Fraz, Muhammad Moazam, et al. "An ensemble classification-based approach applied to retinal blood vessel segmentation." IEEE Transactions on Biomedical Engineering 59.9 (2012): 2538-2548.
. Unzip the dataset into ./retinal_images/original_data/CHASEDB
1.2.4 HRF dataset
. Official website: https://www5.cs.fau.de/research/data/fundus-images/
. The database can be used freely for research purposes. Please cite
  Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg. Robust Vessel Segmentation in Fundus Images. International Journal of Biomedical Imaging, vol. 2013, 2013
. Unzip the dataset into ./retinal_images/original_data/HRF


*1.3 Other datasets
1.3.1 DRISHTI-GS1 (optic disc (OD) and cup (OC))
. Official website: http://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php
. This dataset is free to use. Please cite the following publication when using this dataset: 
  [1]A Comprehensive Retinal Image Dataset for the Assessment of Glaucoma from the Optic Nerve Head Analysis. Sivaswamy J, S. R. Krishnadas, Arunava Chakravarty, Gopal Dutt Joshi, Ujjwal and Tabish Abbas Syed , JSM Biomedical Imaging Data Papers, 2(1):1004, 2015. 
  [2]Drishti-GS: Retinal Image Dataset for Optic Nerve Head(ONH) Segmentation. Sivaswamy J, Krishnadas K. R, Joshi G. D, Jain Madhulika, Ujjwal and Syed Abbas T., IEEE ISBI, Beijing
. Unzip the dataset into ./retinal_images/original_data/Drishti-GS1_files
1.3.2 IDRID （location, segmentation, grading）
. Official website: https://idrid.grand-challenge.org/
. Unzip the dataset into ./retinal_images/original_data/IDRID
1.3.2 octa and DSA vessel mask dataset
. This dataset is in ./retinal_images/prior knowledge

After downloading and unzipping, please run ./retinal_images/preprocessing_of_retinalDataset.py to preprocessing these dataset (including crop roi region and resize to 512x512).


Part 2: Chest X-ray dataset
2.1 NIH Chest X-ray Dataset 
. Official website: https://nihcc.app.box.com/v/ChestXray-NIHCC
. Cite CVPR 2017 paper: Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers.ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471,2017
. Unzip all images into  ./X-ray/CXR8/images
. Run ./X-ray/split_into_train_val_data_list.py to split train and val dataset.
2.2 JSRT and SCR dataset
. JSRT: http://db.jsrt.or.jp/eng.php
. SCR: http://www.isi.uu.nl/Research/Databases/SCR/
. B. van Ginneken, M.B. Stegmann, M. Loog, "Segmentation of anatomical structures in chest radiographs using supervised methods: a comparative study on a public database", Medical Image Analysis, 2006, vol. 10, pp. 19-40.
. J. Shiraishi, S. Katsuragawa, J. Ikezoe, T. Matsumoto, T. Kobayashi, K. Komatsu, M. Matsui, H. Fujita, Y. Kodera, and K. Doi, "Development of a digital image database for chest radiographs with and without a lung nodule: receiver operating characteristic analysis of radiologists' detection of pulmonary nodules", American Journal of Roentgenology, vol. 174, p. 71-74, 2000.
. Original .IMG dataset of JSRT are unzipped into ./X-ray/SCR/oriIMG. And original SCR dataset is unzipped into ./X-ray/SCR/scratch.
2.3 SIIM dataset
. Official website: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation.
. Unzip the dataset into ./x-ray/SIIM-ACR/siim

. Run ./X-ray/preprocessing_of_X-ray.py to preprocess JSRT, SCR and SIIM.

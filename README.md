# Get started

We used repository from the previous research  [[github link]](https://github.com/thedatumorg/TSB-AD).

Due to limitations in the upload size on GitHub, the authors post the datasets in the repa

* TSB-AD-U: https://www.thedatum.org/datasets/TSB-AD-U.zip

* TSB-AD-M: https://www.thedatum.org/datasets/TSB-AD-M.zip

To install TSB-AD from source, you will need the following tools:
- `git`
- `conda` (anaconda or miniconda)

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone https://github.com/pleaseaddhyphens/tsfm-2026.git
```

**Step 2:** Create and activate a `conda` environment named `TSB-AD`.

```bash
conda create -n TSB-AD python=3.11    # Currently we support python>=3.8, up to 3.12
conda activate TSB-AD
```

**Step 3:** Install the dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

If you have problem installing `torch` using pip, try the following:
```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

For instructions on the installation of Foundation Models, please refer to ./TSB-AD/tree/main/TSB_AD/models/README.md.

# Basic Usage CLI

It would run PCA algorithm on the `.\Datasets\TSB-AD-U\001_NAB_id_1_Facility_tr_1007_1st_2014.csv` and calculate all defined metrics.


```bash
python -m TSB_AD.main --AD_Name Sub_PCA
```

Output example:

` {'AUC-PR': 0.7956508501129314, 'AUC-ROC': 0.9059292449548768, 'VUS-PR': 0.8136138874982192, 'VUS-ROC': 0.9357317496173422, 'Standard-F1': 0.8284740249663477, 'PA-F1': 1.0, 'Event-based-F1': 0.9999999999999996, 'R-based-F1': 0.759779845496952, 'Affiliation-F': 0.996308430785461}`


# How to add new anomaly detector

1. Implement new detector class that inherits BaseDetector class (see example at RandomDetector)
2. Define the function in `TSB_AD\model_wrapper.py` that would call new detector class.
3. Add the new class name to  `Unsupervise_AD_Pool` or `Semisupervise_AD_Pool`
4. Add detector's hyperparameters to the `TSB_AD\HP_list.py`. If no hyperparameters required leave the filler (see RandomDetector)



### Detection Algorithms presented in this study

See Implementation in `TSB_AD/models`

We organize the detection algorithms in TSB-AD in the following three categories and arrange these algorithms chronologically within each category.
#### BaseLine Method

| Algorithm    | Description|
|:--|:---------|
|RandomDetector| Randomly assign anomaly score (0,1) to each datapoint from uniform distribution. Implemented as the custom class.|

#### (i) Statistical Method

| Algorithm    | Description|
|:--|:---------|
|(Sub)-MCD|is based on minimum covariance determinant, which seeks to find a subset of all the sequences to estimate the mean and covariance matrix of the subset with minimal determinant. Subsequently, Mahalanobis distance is utilized to calculate the distance from sub-sequences to the mean, which is regarded as the anomaly score.|
|(Sub)-OCSVM|fits the dataset to find the normal data's boundary by maximizing the margin between the origin and the normal samples.|
|(Sub)-LOF|calculates the anomaly score by comparing local density with that of its neighbors.|
|(Sub)-KNN|produces the anomaly score of the input instance as the distance to its $k$-th nearest neighbor.|
|KMeansAD|calculates the anomaly scores for each sub-sequence by measuring the distance to the centroid of its assigned cluster, as determined by the k-means algorithm.|
|CBLOF|is clluster-based LOF, which calculates the anomaly score by first assigning samples to clusters, and then using the distance among clusters as anomaly scores.|
|POLY|detect pointwise anomolies using polynomial approximation. A GARCH method is run on the difference between the approximation and the true value of the dataset to estimate the volatility of each point.|
|(Sub)-IForest|constructs the binary tree, wherein the path length from the root to a node serves as an indicator of anomaly likelihood; shorter paths suggest higher anomaly probability.|
|(Sub)-HBOS|constructs a histogram for the data and uses the inverse of the height of the bin as the anomaly score of the data point.|
|KShapeAD| identifies the normal pattern based on the k-Shape clustering algorithm and computes anomaly scores based on the distance between each sub-sequence and the normal pattern. KShapeAD improves KMeansAD as it relies on a more robust time-series clustering method and corresponds to an offline version of the streaming SAND method.|
|MatrixProfile|identifies anomalies by pinpointing the subsequence exhibiting the most substantial nearest neighbor distance.|
|(Sub)-PCA|projects data to a lower-dimensional hyperplane, with significant deviation from this plane indicating potential outliers.|
|RobustPCA|is built upon PCA and identifies anomalies by recovering the principal matrix.|
|EIF|is an extension of the traditional Isolation Forest algorithm, which removes the branching bias using hyperplanes with random slopes.|
|SR| begins by computing the Fourier Transform of the data, followed by the spectral residual of the log amplitude. The Inverse Fourier Transform then maps the sequence back to the time domain, creating a saliency map. The anomaly score is calculated as the relative difference between saliency map values and their moving averages.|
|COPOD|is a copula-based parameter-free detection algorithm, which first constructs an empirical copula, and then uses it to predict tail probabilities of each given data point to determine its level of extremeness.|
|Series2Graph| converts the time series into a directed graph representing the evolution of subsequences in time. The anomalies are detected using the weight and the degree of the nodes and edges respectively.|
|SAND| identifies the normal pattern based on clustering updated through arriving batches (i.e., subsequences) and calculates each point's effective distance to the normal pattern.|


#### (ii) Neural Network-based Method

| Algorithm    | Description|
|:--|:---------|
|AutoEncoder|projects data to the lower-dimensional latent space and then reconstruct it through the encoding-decoding phase, where anomalies are typically characterized by evident reconstruction deviations.|
|LSTMAD|utilizes Long Short-Term Memory (LSTM) networks to model the relationship between current and preceding time series data, detecting anomalies through discrepancies between predicted and actual values.|
|Donut|is a Variational AutoEncoder (VAE) based method and preprocesses the time series using the MCMC-based missing data imputation technique.|
|CNN|employ Convolutional Neural Network (CNN) to predict the next time stamp on the defined horizon and then compare the difference with the original value.|
|OmniAnomaly|is a stochastic recurrent neural network, which captures the normal patterns of time series by learning their robust representations with key techniques such as stochastic variable connection and planar normalizing flow, reconstructs input data by the representations, and use the reconstruction probabilities to determine anomalies.|
|USAD|is based on adversely trained autoencoders, and the anomaly score is the combination of discriminator and reconstruction loss.|
|AnomalyTransformer|utilizes the `Anomaly-Attention' mechanism to compute the association discrepancy.|
|TranAD|is a deep transformer network-based method, which leverages self-conditioning and adversarial training to amplify errors and gain training stability.|
|TimesNet|is a general time series analysis model with applications in forecasting, classification, and anomaly detection. It features TimesBlock, which can discover the multi-periodicity adaptively and extract the complex temporal variations from transformed 2D tensors by a parameter-efficient inception block.|
|FITS|is a lightweight model that operates on the principle that time series can be manipulated through interpolation in the complex frequency domain.|
|M2N2|uses exponential moving average for trend estimation to detrend data and updates model with normal test instances based on predictions for unsupervised TSAD distribution shifts.

#### (iii) Foundation Model-based Method

| Algorithm    | Description|
|:--|:---------|
|OFA|finetunes pre-trained GPT-2 model on time series data while keeping self-attention and feedforward layers of the residual blocks in the pre-trained language frozen.|
|Lag-Llama|is the first foundation model for univariate probabilistic time series forecasting based on a decoder-only transformer architecture that uses lags as covariates.|
|Chronos|tokenizes time series values using scaling and quantization into a fixed vocabulary and trains the T5 model on these tokenized time series via the cross-entropy loss.|
|Moirai2|is a universal transformer-based time-series forecasting foundation model; in this repository it is used in zero-shot mode, where rolling forecasts are converted into anomaly scores via forecast error.|
|TimesFM|is based on pretraining a decoder-style attention model with input patching, using a large time-series corpus comprising both real-world and synthetic datasets.|
|MOMENT|is pre-trained T5 encoder based on a masked time-series modeling approach.|

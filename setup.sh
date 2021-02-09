pip install -r requirements.txt

# Export your Kaggle username and token to the environment:
# https://github.com/Kaggle/kaggle-api#api-credentials
export KAGGLE_USERNAME=warkingleo2000
export KAGGLE_KEY=xxxxxxxxxxxxxx

mkdir data
kaggle competitions download -c cassava-leaf-disease-classification
unzip cassava-leaf-disease-classification.zip -d data

# pytorch-xla in order to run experiments on TPU
# curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev



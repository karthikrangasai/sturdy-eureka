pip install -r requirements.txt
CURR_DIR=$(pwd)
mkdir data
mkdir outputs
cd data
pip install kaggle
kaggle competitions download -c chaii-hindi-and-tamil-question-answering
mkdir -p chaii-hindi-and-tamil-question-answering
unzip -o chaii-hindi-and-tamil-question-answering.zip -d chaii-hindi-and-tamil-question-answering
rm chaii-hindi-and-tamil-question-answering.zip
cd $CURR_DIR
python src/create_splits.py
all:

install:
	conda create -n e2e.seq2seq python=3.6.8
	conda activate e2e.seq2seq
	conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
	pip install -r requirements.txt

tar:
	tar cvfz clovacall_testdata_model.tar.gz data/zeroth_korean/test_data_01 models

untar:
	gdown --id 1T5n7hihhlICV9rKu0xTxMccApkI5tRit --output clovacall_testdata_model.tar.gz
	tar xvfz clovacall_testdata_model.tar.gz

data:
	

train:
	cd script.zeroth_korean && ./run_las_asr_trainer.sh

train.trim:
	cd script.zeroth_korean && ./run_las_asr_trainer.trimmed.sh

test:
	cd script.zeroth_korean && ./run_las_asr_decode.sh

test.trim:
	cd script.zeroth_korean && ./run_las_asr_decode.trimmed.sh

train.ksponspeech:
	cd script.ksponspeech && ./run_las_asr_trainer.sh

server:
	mkdir -p api/audio
	cd api && CUDA_VISIBLE_DEVICES= python api_server.py

path:
	export PYTHONPATH=${PYTHONPATH}:${PWD}/las.pytorch

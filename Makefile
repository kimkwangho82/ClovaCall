all:

install:
	conda create -n e2e.seq2seq python=3.6.8
	conda activate e2e.seq2seq
	conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
	pip install -r requirements.txt

train:
	cd script.zeroth_korean && ./run_las_asr_trainer.sh

train.trim:
	cd script.zeroth_korean && ./run_las_asr_trainer.trimmed.sh

test:
	cd script.zeroth_korean && ./run_las_asr_decode.sh

test.trim:
	cd script.zeroth_korean && ./run_las_asr_decode.trimmed.sh
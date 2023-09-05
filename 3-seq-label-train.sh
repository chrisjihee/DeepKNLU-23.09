python 3-seq-label.py train --data-name klue-ner --seq-len 256 --batch-size 64 --num-save 5 --epochs 3 --pretrained pretrained/KcBERT --device 0
python 3-seq-label.py train --data-name klue-ner --seq-len 256 --batch-size 64 --num-save 5 --epochs 3 --pretrained pretrained/KPF-BERT --device 1
python 3-seq-label.py train --data-name klue-ner --seq-len 256 --batch-size 64 --num-save 5 --epochs 3 --pretrained pretrained/KLUE-BERT --device 2
python 3-seq-label.py train --data-name klue-ner --seq-len 256 --batch-size 64 --num-save 5 --epochs 3 --pretrained pretrained/KoELECTRA --device 3

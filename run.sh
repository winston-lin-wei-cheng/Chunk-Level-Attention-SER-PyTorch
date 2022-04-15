#python training.py -iter 5000 -batch 128 -emo Act -atten RnnAttenVec;
#sleep 10;
#python training.py -iter 5000 -batch 128 -emo Dom -atten RnnAttenVec;
#sleep 10;
#python training.py -iter 5000 -batch 128 -emo Val -atten RnnAttenVec;
#sleep 10;
#python training.py -iter 5000 -batch 128 -emo Act -atten SelfAttenVec;
#sleep 10;
#python training.py -iter 5000 -batch 128 -emo Dom -atten SelfAttenVec;
#sleep 10;
#python training.py -iter 5000 -batch 128 -emo Val -atten SelfAttenVec;
#sleep 10;


python testing.py -iter 5000 -batch 128 -emo Act -atten RnnAttenVec;
python testing.py -iter 5000 -batch 128 -emo Dom -atten RnnAttenVec;
python testing.py -iter 5000 -batch 128 -emo Val -atten RnnAttenVec;
python testing.py -iter 5000 -batch 128 -emo Act -atten SelfAttenVec;
python testing.py -iter 5000 -batch 128 -emo Dom -atten SelfAttenVec;
python testing.py -iter 5000 -batch 128 -emo Val -atten SelfAttenVec;


python post_show.py --config ./configs/CoRe.json --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1
python post_show.py --config ./configs/CoRe.json --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 0
python post_show.py --config ./configs/CoRe.json --default_config ./configs/default_config.json --fold 0 --al 0 --ms 1
python post_show.py --config ./configs/CoRe.json --default_config ./configs/default_config.json --fold 0 --al 0 --ms 0

python post_show.py --config ./configs/Early.json --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1
python post_show.py --config ./configs/Early.json --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 0
python post_show.py --config ./configs/Early.json --default_config ./configs/default_config.json --fold 0 --al 0 --ms 1
python post_show.py --config ./configs/Early.json --default_config ./configs/default_config.json --fold 0 --al 0 --ms 0

python post_show.py --config ./configs/MidLate.json --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1
python post_show.py --config ./configs/MidLate.json --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 0
python post_show.py --config ./configs/MidLate.json --default_config ./configs/default_config.json --fold 0 --al 0 --ms 1
python post_show.py --config ./configs/MidLate.json --default_config ./configs/default_config.json --fold 0 --al 0 --ms 0

python post_show.py --config ./configs/UniEEG.json --default_config ./configs/default_config.json --fold 0
python post_show.py --config ./configs/UniEOG.json --default_config ./configs/default_config.json --fold 0

python train.py --config ./configs/CoRe_incomplete.json --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1 --incboth 100 --inceog 0 --inceeg 0


python train.py --config ./configs/CoRe.json    --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1
python train.py --config ./configs/Early.json   --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1
python train.py --config ./configs/MidLate.json --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1
python train.py --config ./configs/UniEEG.json  --default_config ./configs/default_config.json --fold 0
python train.py --config ./configs/UniEOG.json  --default_config ./configs/default_config.json --fold 0

python post_test.py --config ./configs/CoRe.json    --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1
python post_test.py --config ./configs/Early.json   --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1
python post_test.py --config ./configs/MidLate.json --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1
python post_test.py --config ./configs/UniEEG.json  --default_config ./configs/default_config.json --fold 0
python post_test.py --config ./configs/UniEOG.json  --default_config ./configs/default_config.json --fold 0

python post_test.py --config ./configs/CoRe.json    --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1 --noisy
python post_test.py --config ./configs/Early.json   --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1 --noisy
python post_test.py --config ./configs/MidLate.json --default_config ./configs/default_config.json --fold 0 --al 0.1 --ms 1 --noisy
python post_test.py --config ./configs/UniEEG.json  --default_config ./configs/default_config.json --fold 0 --noisy
python post_test.py --config ./configs/UniEOG.json  --default_config ./configs/default_config.json --fold 0 --noisy


#remove all the unused files and group some norm or train/val files together to organise it. Then you should be good for an init release with a readme.
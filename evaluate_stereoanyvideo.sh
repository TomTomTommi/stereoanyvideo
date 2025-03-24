#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

# evaluate on [sintel, dynamicreplica， infinigensv, vkitti2] using sceneflow checkpoint

python ./evaluation/evaluate.py --config-name eval_sintel_final \
MODEL.model_name=StereoAnyVideoModel \
MODEL.StereoAnyVideoModel.model_weights=./checkpoints/StereoAnyVideo_SF.pth

python ./evaluation/evaluate.py --config-name eval_dynamic_replica \
MODEL.model_name=StereoAnyVideoModel \
MODEL.StereoAnyVideoModel.model_weights=./checkpoints/StereoAnyVideo_SF.pth

python ./evaluation/evaluate.py --config-name eval_infinigensv \
MODEL.model_name=StereoAnyVideoModel \
MODEL.StereoAnyVideoModel.model_weights=./checkpoints/StereoAnyVideo_SF.pth

python ./evaluation/evaluate.py --config-name eval_vkitti2 \
MODEL.model_name=StereoAnyVideoModel \
MODEL.StereoAnyVideoModel.model_weights=./checkpoints/StereoAnyVideo_SF.pth



# evaluate on [sintel, kittidepth, southkensingtonSV] using mixed checkpoint

python ./evaluation/evaluate.py --config-name eval_sintel_final \
MODEL.model_name=StereoAnyVideoModel \
MODEL.StereoAnyVideoModel.model_weights=./checkpoints/StereoAnyVideo_MIX.pth

python ./evaluation/evaluate.py --config-name eval_kittidepth \
MODEL.model_name=StereoAnyVideoModel \
MODEL.StereoAnyVideoModel.model_weights=./checkpoints/StereoAnyVideo_MIX.pth

python ./evaluation/evaluate.py --config-name eval_southkensington \
MODEL.model_name=StereoAnyVideoModel \
MODEL.StereoAnyVideoModel.model_weights=./checkpoints/StereoAnyVideo_SF.pth
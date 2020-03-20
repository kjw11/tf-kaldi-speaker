#!/bin/bash
. ./cmd.sh
. ./path.sh
set -e

root=/work9/cslt/kangjiawen/122019-tf-kaldi/experiments/vox-resnet34-sp-sm
data=$root/data
exp=$root/exp
nnetdir=$exp/nnet_resnet
mfccdir=$root/mfcc
vaddir=$root/mfcc
fbankdir=$root/fbank

kaldi_sitw=/work9/cslt/kangjiawen/kaldi/egs/sitw
voxceleb1_root=/nfs/corpus0/data/corpora/database/speech/sid/VoxCeleb/voxceleb1
voxceleb2_root=/nfs/corpus0/data/corpora/database/speech/sid/VoxCeleb/voxceleb2
sitw_root=/work8/lilt/database/SITW
musan_root=/work9/cslt/kangjiawen/database/musan
rirs_root=/work9/cslt/kangjiawen/database/RIRS_NOISES
eval_trials_core=$data/eval_test/trials/core-core.lst

config=nnet_conf/resnet34_stat_pooling_softmax.json
checkpoint='last'
stage=1

if [ $stage -le -1 ]; then
    # link the directories
    rm -fr utils steps sid local
    ln -s $kaldi_sitw/v2/utils ./
    ln -s $kaldi_sitw/v2/steps ./
    ln -s $kaldi_sitw/v2/sid ./
    ln -s $kaldi_sitw/v1/local ./
    exit 1
fi

if [ $stage -le 0 ]; then
  #  make sitw data
  local/make_sitw.sh $sitw_root $data
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sitw_eval_enroll sitw_eval_test sitw_dev_enroll sitw_dev_test; do  
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      $data/${name} $exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      $data/${name} $exp/make_vad $vaddir
    utils/fix_data_dir.sh $data/${name}
    steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank.conf --nj 40 --cmd "$train_cmd" \
      $data/${name} exp/make_fbank $fbankdir
    utils/fix_data_dir.sh $data/${name}
  done
fi

if [ $stage -le 8 ]; then
  # Extract the embeddings
  for name in sitw_eval_enroll sitw_dev_enroll sitw_eval_test sitw_dev_test; do
    nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 40 \ 
      --use-gpu true --checkpoint $checkpoint --stage 0 --min-chunk-size 10 \
      --chunk-size 10000 --normalize false --node "tdnn6_dense" \
      $nnetdir $data/${name}  $nnetdir/xvectors_${name}
  done
fi


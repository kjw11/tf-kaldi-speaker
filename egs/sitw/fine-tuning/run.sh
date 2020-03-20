#!/bin/bash
. ./cmd.sh
. ./path.sh
set -e

root=/work9/cslt/kangjiawen/122019-tf-kaldi/experiments/vox-resnet34-sp-sm
data=$root/data
exp=$root/exp
nnetdir=$exp/nnet_resnet_finetuning
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

config=nnet_conf/resnet34_stat_pooling_arcsoftmax.jso
pretrain_nnet=nnetdir=$exp/nnet_resnet
checkpoint='last'
stage=1

if [ $stage -le -1 ]; then
    # link the directories
    rm -fr utils steps sid conf local
    ln -s $kaldi_sitw/v2/utils ./
    ln -s $kaldi_sitw/v2/steps ./
    ln -s $kaldi_sitw/v2/sid ./
    ln -s $kaldi_sitw/v1/local ./
    exit 1
fi

if [ $stage -le 0 ]; then
  #  make voxceleb and sitw data
  local/make_voxceleb1.pl $voxceleb1_root $data
  local/make_voxceleb2.pl $voxceleb2_root dev $data/voxceleb2_train
  utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb1
  local/make_sitw.sh $sitw_root $data
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sitw_eval_enroll sitw_eval_test sitw_dev_enroll sitw_dev_test train; do  
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

if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $data/train/utt2num_frames > $data/train/reco2dur


  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  local/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    $data/train $data/train_reverb
  cp $data/train/vad.scp $data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" $data/train_reverb $data/train_reverb.new
  rm -rf $data/train_reverb
  mv $data/train_reverb.new $data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh $musan_root $data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh $data/musan_${name}
    mv $data/musan_${name}/utt2dur $data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python3 steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "$data/musan_noise" $data/train $data/train_noise
  # Augment with musan_music
  python3 steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "$data/musan_music" $data/train $data/train_music
  # Augment with musan_speech
  python3 steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "$data/musan_speech" $data/train $data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh $data/train_aug $data/train_reverb $data/train_noise $data/train_music $data/train_babble
fi

if [ $stage -le 3 ]; then
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh $data/train_aug 1000000 $data/train_aug_1m
  utils/fix_data_dir.sh $data/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_fbank.sh --fbank-config conf/fbank.conf --nj 40 --cmd "$train_cmd" \
    $data/train_aug_1m exp/make_fbank $fbankdir

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh $data/train_combined $data/train_aug_1m $data/train
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    $data/train_combined $data/train_combined_no_sil $exp/train_combined_no_sil
  utils/fix_data_dir.sh $data/train_combined_no_sil
fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv $data/train_combined_no_sil/utt2num_frames $data/train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data/train_combined_no_sil/utt2num_frames.bak > $data/train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl $data/train_combined_no_sil/utt2num_frames $data/train_combined_no_sil/utt2spk > $data/train_combined_no_sil/utt2spk.new
  mv $data/train_combined_no_sil/utt2spk.new $data/train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh $data/train_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' $data/train_combined_no_sil/spk2utt > $data/train_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data/train_combined_no_sil/spk2num | utils/filter_scp.pl - $data/train_combined_no_sil/spk2utt > $data/train_combined_no_sil/spk2utt.new
  mv $data/train_combined_no_sil/spk2utt.new $data/train_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/train_combined_no_sil/spk2utt > $data/train_combined_no_sil/utt2spk

  utils/filter_scp.pl $data/train_combined_no_sil/utt2spk $data/train_combined_no_sil/utt2num_frames > $data/train_combined_no_sil/utt2num_frames.new
  mv $data/train_combined_no_sil/utt2num_frames.new $data/train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh $data/train_combined_no_sil
fi

if [ $stage -le 6 ]; then
  # Split the validation set
  num_heldout_spks=64
  num_heldout_utts_per_spk=20
  mkdir -p $data/train_combined_no_sil/train/ $data/train_combined_no_sil/valid/

  sed 's/-noise//' $data/train_combined_no_sil/utt2spk | sed 's/-music//' | sed 's/-babble//' | sed 's/-reverb//' |\
    paste -d ' ' $data/train_combined_no_sil/utt2spk - | cut -d ' ' -f 1,3 > $data/train_combined_no_sil/utt2uniq

  utils/utt2spk_to_spk2utt.pl $data/train_combined_no_sil/utt2uniq > $data/train_combined_no_sil/uniq2utt
  cat $data/train_combined_no_sil/utt2spk | utils/apply_map.pl -f 1 $data/train_combined_no_sil/utt2uniq |\
    sort | uniq > $data/train_combined_no_sil/utt2spk.uniq

  utils/utt2spk_to_spk2utt.pl $data/train_combined_no_sil/utt2spk.uniq > $data/train_combined_no_sil/spk2utt.uniq
  python $TF_KALDI_ROOT/misc/tools/sample_validset_spk2utt.py $num_heldout_spks $num_heldout_utts_per_spk $data/train_combined_no_sil/spk2utt.uniq > $data/train_combined_no_sil/valid/spk2utt.uniq

  cat $data/train_combined_no_sil/valid/spk2utt.uniq | utils/apply_map.pl -f 2- $data/train_combined_no_sil/uniq2utt > $data/train_combined_no_sil/valid/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/train_combined_no_sil/valid/spk2utt > $data/train_combined_no_sil/valid/utt2spk
  cp $data/train_combined_no_sil/feats.scp $data/train_combined_no_sil/valid
  utils/filter_scp.pl $data/train_combined_no_sil/valid/utt2spk $data/train_combined_no_sil/utt2num_frames > $data/train_combined_no_sil/valid/utt2num_frames
  utils/fix_data_dir.sh $data/train_combined_no_sil/valid

  utils/filter_scp.pl --exclude $data/train_combined_no_sil/valid/utt2spk $data/train_combined_no_sil/utt2spk > $data/train_combined_no_sil/train/utt2spk
  utils/utt2spk_to_spk2utt.pl $data/train_combined_no_sil/train/utt2spk > $data/train_combined_no_sil/train/spk2utt
  cp $data/train_combined_no_sil/feats.scp $data/train_combined_no_sil/train
  utils/filter_scp.pl $data/train_combined_no_sil/train/utt2spk $data/train_combined_no_sil/utt2num_frames > $data/train_combined_no_sil/train/utt2num_frames
  utils/fix_data_dir.sh $data/train_combined_no_sil/train

  awk -v id=0 '{print $1, id++}' $data/train_combined_no_sil/train/spk2utt > $data/train_combined_no_sil/train/spklist
    
  # top 200k
  sort -n -k 2 $data/train_combined_no_sil/utt2num_frames | tail -n 200000 > $data/train_combined_no_sil/train_200k.list
  utils/subset_data_dir.sh --utt-list $data/train_combined_no_sil/train_200k.list \
    $data/train_combined $data/train_combined_200k
fi


if [ $stage -le 7 ]; then
  # Training a resnet network
  nnet/run_finetune_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false --checkpoint 'last' \
     $config \
     $data/train_combined_no_sil/train $data/train_combined_no_sil/train/spklist \
     $data/train_combined_no_sil/valid $data/train_combined_no_sil/train/spklist \
     $pretrain_nnet $nnetdirr
  exit 1
fi

if [ $stage -le 8 ]; then
  # Extract the embeddings
  for name in sitw_eval_enroll sitw_dev_enroll sitw_eval_test sitw_dev_test train_combined_200k; do
    nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj 40 \ 
      --use-gpu true --checkpoint $checkpoint --stage 0 --min-chunk-size 10 \
      --chunk-size 10000 --normalize false --node "tdnn6_dense" \
      $nnetdir $data/${name}  $nnetdir/xvectors_${name}
  done
fi

if [ $stage -le 10 ]; then
  # Compute the mean.vec used for centering.
  $train_cmd $nnetdir/xvectors_train_comb/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_train_comb/xvector.scp \
    $nnetdir/xvectors_train_comb/mean.vec || exit 1;

  # Use LDA to decrease the dimensionality prior to PLDA.
  lda_dim=128
  $train_cmd $nnetdir/xvectors_train_comb/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_train_comb/xvector.scp ark:- |" \
    ark:$data/train_comb/utt2spk $nnetdir/xvectors_train_comb/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnetdir/xvectors_train_comb/log/plda.log \
    ivector-compute-plda ark:$data/train_comb/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_train_comb/xvector.scp ark:- | transform-vec $nnetdir/xvectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_train_comb/plda || exit 1;
fi

if [ $stage -le 11 ]; then
  # Compute PLDA scores for Vox-Celeb eval core trials
  $train_cmd $nnetdir/scores/log/voxceleb_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_train_comb/plda - |" \
    "ark:ivector-mean ark:$data/eval_enroll/spk2utt scp:$nnetdir/xvectors_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_train_comb/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_train_comb/mean.vec scp:$nnetdir/xvectors_eval_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$eval_trials_core' | cut -d\  --fields=1,2 |" $nnetdir/scores/voxceleb_eval_scores || exit 1;

  echo -e "\nVox-Celeb Eval Core:";
#  eer=$(paste $eval_trials_core $nnetdir/scores/cnceleb_eval_scores | awk '{print $6, $3}' > 123)
  eer=$(paste $eval_trials_core $nnetdir/scores/voxceleb_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnetdir/scores/voxceleb_eval_scores $eval_trials_core 2>/dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnetdir/scores/voxceleb_eval_scores $eval_trials_core 2>/dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"

  end_time=$(date +%s)
  cost_time=$[ $start_time-$end_time ]
  echo " cost time : $(($cost_time/3600))h $(($cost_time%3600/60))m"
fi

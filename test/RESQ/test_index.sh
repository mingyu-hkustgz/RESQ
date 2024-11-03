source set.sh

for data in "${datasets[@]}"; do
  if [ $data == "msong" ]; then
    B=128
  elif [ $data == "deep1M" ]; then
    B=128
  elif [ $data == "gist" ]; then
    B=128
  elif [ $data == "tiny5m" ]; then
    B=128
  elif [ $data == "word2vec" ]; then
    B=256
  elif [ $data == "sift" ]; then
    B=64
  elif [ $data == "glove2.2m" ]; then
    B=256
  elif [ $data == "OpenAI-1536" ]; then
    B=512
  elif [ $data == "OpenAI-3072" ]; then
    B=512
  elif [ $data == "msmarc-small" ]; then
    B=512
  elif [ $data == "yt1m" ]; then
    B=512
  fi

  log_file="./results/time-log/${data}/ResQ-PCA-Train-time.log"
  start_time=$(date +%s)
  python ./script/RESQ/pca.py -d ${data} -b ${B}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "PCA Train time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/ResQ-IVF-Train-time.log"
  start_time=$(date +%s)
  python ./script/RESQ/ivf.py -d ${data} -b ${B}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Resq IVF Train time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/ResQ-Encode-time.log"
  start_time=$(date +%s)
  python ./script/RESQ/resq.py -d ${data} -b ${B}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Resq Encode time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/ResQ-Index-time.log"
  start_time=$(date +%s)
  ./cmake-build-debug/src/res_index -d ${data} -s "./DATA/${data}/" -b ${B}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Resq IVF Index time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/ResQ-Split-time.log"
  start_time=$(date +%s)
  ./cmake-build-debug/src/res_split_index -d ${data} -s "./DATA/${data}/" -b ${B}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Resq IVF Split time: ${duration}(s)" | tee -a ${log_file}

done

source set.sh

for data in "${datasets[@]}"; do
  if [ $data == "msong" ]; then
    B=448
  elif [ $data == "deep1M" ]; then
    B=256
  elif [ $data == "word2vec" ]; then
    B=320
  elif [ $data == "glove2.2m" ]; then
    B=320
  elif [ $data == "tiny5m" ]; then
    B=384
  elif [ $data == "sift" ]; then
    B=128
  elif [ $data == "gist" ]; then
    B=960
  elif [ $data == "OpenAI-1536" ]; then
    B=1536
  elif [ $data == "OpenAI-3072" ]; then
    B=3072
  elif [ $data == "msmarc-small" ]; then
    B=1024
  elif [ $data == "msmarc10m" ]; then
    B=512
  elif [ $data == "yt1m" ]; then
    B=1024
  fi

  log_file="./results/time-log/${data}/RabitQ-IVF-Train-time.log"
  start_time=$(date +%s)
  python ./script/RaBitQ/ivf.py -d ${data}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "RabitQ IVF Train time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/RabitQ-Encode-time.log"
  start_time=$(date +%s)
  python ./script/RaBitQ/rabitq.py -d ${data}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "RabitQ Encode time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/RabitQ-IVF-Index-time.log"
  start_time=$(date +%s)
  ./cmake-build-debug/src/rabit_index -d ${data} -s "./DATA/${data}/" -b ${B}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "RabitQ IVF Index time: ${duration}(s)" | tee -a ${log_file}

done

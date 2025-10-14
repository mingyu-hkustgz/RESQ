source set.sh

for data in "${datasets[@]}"; do
  if [ $data == "msmarc10m" ]; then
    B=320
    C=4096
  elif [ $data == "gist" ]; then
    B=320
    C=4096
  elif [ $data == "msmarc20m" ]; then
    B=320
    C=4096
  elif [ $data == "msmarc40m" ]; then
    B=320
    C=8192
  elif [ $data == "msmarc60m" ]; then
    B=320
    C=8912
  elif [ $data == "msmarc80m" ]; then
    B=320
    C=16384
  elif [ $data == "msmarc" ]; then
    B=320
    C=16384
  fi

  log_file="./results/time-log/${data}/ResQ-PCA-Train-time.log"
  start_time=$(date +%s)
  python ./script/RESQ/pca-large.py -d ${data} -b ${B}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "PCA Train time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/ResQ-IVF-Train-time.log"
  start_time=$(date +%s)
  python ./script/RESQ/ivf-large.py -d ${data} -b ${B} -c ${C}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Resq IVF Train time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/ResQ-Encode-time.log"
  start_time=$(date +%s)
  python ./script/RESQ/resq-large.py -d ${data} -b ${B} -c ${C}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Resq Encode time: ${duration}(s)" | tee -a ${log_file}

#  log_file="./results/time-log/${data}/ResQ-Index-time.log"
#  start_time=$(date +%s)
#  ./build/src/res_index -d ${data} -s "./DATA/${data}/" -b ${B} -c ${C}
#  end_time=$(date +%s)
#  duration=$((end_time - start_time))
#  echo "Resq IVF Index time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/ResQ-Split-time.log"
  start_time=$(date +%s)
  ./build/src/res_split_index -d ${data} -s "./DATA/${data}/" -b ${B} -c ${C}
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "Resq IVF Split time: ${duration}(s)" | tee -a ${log_file}

  LOG_FILE="./results/space-log/${data}/ResQ-Split-Index.log"
  FILE_SIZE=$(stat -c%s "./DATA/${data}/ivf_split${C}_B${B}.index")
  echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO: File '$FILE_PATH' size: $FILE_SIZE bytes" >> "$LOG_FILE"

  K=20
  res="./results/recall@${K}/${data}/"
  ./build/src/res_split_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B} -c ${C}

done

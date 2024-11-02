source set.sh

for data in "${datasets[@]}"; do
  for K in {20,100}; do
    if [ $data == "msong" ]; then
      B=448
      P=5
    elif [ $data == "deep1M" ]; then
      B=256
      P=25
    elif [ $data == "word2vec" ]; then
      B=320
      P=15
    elif [ $data == "glove2.2m" ]; then
      B=320
      P=15
    elif [ $data == "tiny5m" ]; then
      B=384
      P=25
    elif [ $data == "sift" ]; then
      B=128
      P=8
    elif [ $data == "gist" ]; then
      B=960
      P=25
    elif [ $data == "OpenAI-1536" ]; then
      B=1536
      P=30
    elif [ $data == "OpenAI-3072" ]; then
      B=3072
      P=30
    elif [ $data == "msmarc-small" ]; then
      B=1024
      P=30
    fi
    res="${result_path}/recall@${K}/${data}/"
    ./cmake-build-debug/src/rabit_scan_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B}
    ./cmake-build-debug/src/rabit_fast_scan_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B}

    for (( probe=${P}; probe<=${P}*20; probe+=${P}))
    do
      echo "disk probe ${probe}"
      ./cmake-build-debug/src/rabit_fast_disk_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B} -p ${probe}
    done
  done
done

source set.sh

for data in "${datasets[@]}"; do
  for K in {20,100}; do
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
    fi
    res="${result_path}/recall@${K}/${data}/"
    ./cmake-build-debug/src/rabit_scan_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B}
    ./cmake-build-debug/src/rabit_fast_scan_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B}
    ./cmake-build-debug/src/rabit_fast_disk_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B}
  done
done

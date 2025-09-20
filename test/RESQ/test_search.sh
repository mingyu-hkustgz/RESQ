source set.sh

for data in "${datasets[@]}"; do
  for K in {20,100}; do
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
    elif [ $data == "msmarc10m" ]; then
      B=512
    elif [ $data == "yt1m" ]; then
      B=512
    fi
    res="${result_path}/recall@${K}/${data}/"
    ./cmake-build-debug/src/res_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B} -t 1
    ./cmake-build-debug/src/res_split_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B} -t 1
  done
done

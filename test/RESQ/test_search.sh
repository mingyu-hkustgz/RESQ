source set.sh

for data in "${datasets[@]}"; do
  for K in {20,100}; do
    if [ $data == "msong" ]; then
      B=128
      P=5
    elif [ $data == "deep1M" ]; then
      B=128
      P=25
    elif [ $data == "gist" ]; then
      B=128
      P=15
    elif [ $data == "tiny5m" ]; then
      B=128
      P=25
    elif [ $data == "word2vec" ]; then
      B=256
      P=15
    elif [ $data == "sift" ]; then
      B=64
      P=8
    elif [ $data == "glove2.2m" ]; then
      B=256
      P=15
    elif [ $data == "OpenAI-1536" ]; then
      B=512
      P=30
    elif [ $data == "OpenAI-3072" ]; then
      B=512
      P=30
    elif [ $data == "msmarc-small" ]; then
      B=512
      P=30
    elif [ $data == "yt1m" ]; then
      B=512
      P=30
    fi
    res="${result_path}/recall@${K}/${data}/"
    ./build/src/res_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B}
    ./build/src/res_split_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B}

  done
done

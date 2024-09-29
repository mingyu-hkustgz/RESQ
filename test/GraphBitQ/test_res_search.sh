source set.sh

for data in "${datasets[@]}"; do
  for K in {20,100}; do
    if [ $data == "msong" ]; then
      B=128
    elif [ $data == "deep1M" ]; then
      B=128
    elif [ $data == "word2vec" ]; then
      B=256
    elif [ $data == "glove2.2m" ]; then
      B=256
    elif [ $data == "tiny5m" ]; then
      B=128
    elif [ $data == "sift" ]; then
      B=64
    elif [ $data == "gist" ]; then
      B=256
    fi
    res="${result_path}/recall@${K}/${data}/"
    ./cmake-build-debug/src/graph_res_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B}
  done
done

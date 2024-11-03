source set.sh

for K in {20,100}; do
  for data in "${datasets[@]}"; do
    echo "Searching - ${data}"

    if [ $data == "OpenAI-1536" ]; then
      efSearch=50
    elif [ $data == "msong" ]; then
      efSearch=20
    elif [ $data == "OpenAI-3072" ]; then
      efSearch=50
    elif [ $data == "glove2.2m" ]; then
      efSearch=50
    elif [ $data == "gist" ]; then
      efSearch=100
    elif [ $data == "deep1M" ]; then
      efSearch=50
    elif [ $data == "msmarc-small" ]; then
      efSearch=50
    elif [ $data == "deep100M" ]; then
      efSearch=50
    fi

    data_path=${store_path}/${data}
    result_path="./results/recall@${K}/${data}"
    query="${data_path}/${data}_query.fvecs"
    gnd="${data_path}/${data}_groundtruth.ivecs"
    ef=500
    M=16
    index="${data_path}/${data}_ef${ef}_M${M}.index"

    res="${result_path}/${data}_hnsw_naive.log"
    ./cmake-build-debug/src/search_hnsw -i ${index} -q ${query} -g ${gnd} -r ${res}  -k ${K} -s ${efSearch}
  done
done

source set.sh

for data in "${datasets[@]}"; do
    for K in {20,100}; do
      if [ $data == "msong" ]; then
        B=448
      elif [ $data == "deep1M" ]; then
        B=128
      elif [ $data == "word2vec" ]; then
        B=320
      elif [ $data == "msmarc" ]; then
        B=512
      elif [ $data == "tiny5m" ]; then
        B=128
      elif [ $data == "OpenAI-1536" ]; then
        B=512
      elif [ $data == "OpenAI-3072" ]; then
        B=512
      elif [ $data == "gist" ]; then
        B=128
      elif [ $data == "sift" ]; then
        B=64
      fi
      result_path="./results/recall@${K}/${data}/"
      ./cmake-build-debug/src/res_search_hnsw -d ${data} -s "./DATA/${data}/" -b ${B} -r ${result_path}
    done
done
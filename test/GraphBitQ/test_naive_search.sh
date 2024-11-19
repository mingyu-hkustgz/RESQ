source set.sh

for data in "${datasets[@]}"; do
  for K in {20,100}; do
    res="${result_path}/recall@${K}/${data}/"
    ./cmake-build-debug/src/graph_naive_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/"
  done
done
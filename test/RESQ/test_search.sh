source set.sh

for data in "${datasets[@]}"; do
  for K in {20,100}; do
    if [ $data == "sift" ]; then
      B=128
    elif [ $data == "gist" ]; then
      B=960
    fi
    res="${result_path}/recall@${K}/${data}/"
    ./cmake-build-debug/src/itq_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B}
  done
done

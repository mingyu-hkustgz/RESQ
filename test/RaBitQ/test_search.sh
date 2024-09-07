source set.sh

for data in "${datasets[@]}"; do
  for K in {20,100}; do
    if [ $data == "sift" ]; then
      B=128
    elif [ $data == "gist" ]; then
      B=960
    elif [ $data == "pgist" ]; then
      B=512
    elif [ $data == "ppgist" ]; then
      B=256
    elif [ $data == "pppgist" ]; then
      B=128
    fi
    res="${result_path}/recall@${K}/${data}/"
    ./cmake-build-debug/src/rabit_search -d ${data} -k ${K} -r ${res} -s "./DATA/${data}/" -b ${B}
  done
done

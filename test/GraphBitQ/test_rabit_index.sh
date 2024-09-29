source set.sh

for data in "${datasets[@]}"; do
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
  fi
  python ./script/RaBitQ/rabitq-o.py -d ${data}
  ./cmake-build-debug/src/graph_rabit_index -d ${data} -s "./DATA/${data}/" -b ${B}

done

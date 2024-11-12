source set.sh

for data in "${datasets[@]}"; do
  if [ $data == "msong" ]; then
    B=448
  elif [ $data == "deep1M" ]; then
    B=256
  elif [ $data == "word2vec" ]; then
    B=320
  elif [ $data == "msmarc" ]; then
    B=1024
  elif [ $data == "tiny5m" ]; then
    B=384
  elif [ $data == "OpenAI-1536" ]; then
    B=1536
  elif [ $data == "OpenAI-3072" ]; then
    B=3072
  elif [ $data == "gist" ]; then
    B=960
  fi
  python ./script/RaBitQ/rabitq-o.py -d ${data}
  ./cmake-build-debug/src/graph_rabit_index -d ${data} -s "./DATA/${data}/" -b ${B}

done
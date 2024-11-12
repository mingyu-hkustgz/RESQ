source set.sh

for data in "${datasets[@]}"; do
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
  fi
    python ./script/RESQ/resq-o.py -d ${data} -b ${B}
  ./cmake-build-debug/src/graph_res_index -d ${data} -s "./DATA/${data}/" -b ${B}

done
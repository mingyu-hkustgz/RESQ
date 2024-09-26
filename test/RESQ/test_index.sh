source set.sh

for data in "${datasets[@]}"; do
  if [ $data == "msong" ]; then
    B=128
  elif [ $data == "deep1M" ]; then
    B=128
  elif [ $data == "gist" ]; then
    B=128
  elif [ $data == "tiny5m" ]; then
    B=128
  elif [ $data == "word2vec" ]; then
    B=256
  elif [ $data == "sift" ]; then
    B=64
  elif [ $data == "glove2.2m" ]; then
    B=256
  fi
  python ./script/RESQ/pca.py -d ${data} -b ${B}
  python ./script/RESQ/ivf.py -d ${data} -b ${B}
  python ./script/RESQ/resq.py -d ${data} -b ${B}
  ./cmake-build-debug/src/res_index -d ${data} -s "./DATA/${data}/" -b ${B}
  ./cmake-build-debug/src/res_split_index -d ${data} -s "./DATA/${data}/" -b ${B}

done

source set.sh

for data in "${datasets[@]}"; do
  if [ $data == "sift" ]; then
    B=64
  elif [ $data == "gist" ]; then
    B=128
  fi
  python ./script/RESQ/pca.py -d ${data} -b ${B}
  python ./script/RESQ/ivf.py -d ${data} -b ${B}
  python ./script/RESQ/RESQ.py -d ${data} -b ${B}
  ./cmake-build-debug/src/res_index -d ${data} -s "./DATA/${data}/" -b ${B}

done

source set.sh

for data in "${datasets[@]}"; do
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
  python ./script/RaBitQ/ivf.py -d ${data}
  python ./script/RaBitQ/rabitq.py -d ${data}
  ./cmake-build-debug/src/rabit_index -d ${data} -s "./DATA/${data}/" -b ${B}

done

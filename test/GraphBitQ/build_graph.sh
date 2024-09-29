source set.sh

for data in "${datasets[@]}"; do
  if [ $data == "msong" ]; then
    K=400
    L=400
    iter=12
    S=15
    R=100
  elif [ $data == "deep1M" ]; then
    K=400
    L=400
    iter=12
    S=15
    R=100
  elif [ $data == "word2vec" ]; then
    K=400
    L=400
    iter=15
    S=15
    R=200
  elif [ $data == "glove2.2m" ]; then
    K=400
    L=400
    iter=15
    S=15
    R=200
  elif [ $data == "tiny5m" ]; then
    K=400
    L=400
    iter=12
    S=15
    R=100
  elif [ $data == "sift" ]; then
    K=400
    L=400
    iter=12
    S=15
    R=100
  elif [ $data == "gist" ]; then
    K=400
    L=400
    iter=12
    S=15
    R=100
  fi

  ./cmake-build-debug/src/test_nndescent "./DATA/${data}/${data}_base.fvecs" "./DATA/${data}/${data}_${K}.graph" $K $L $iter $S $R
  ./cmake-build-debug/src/test_ssg_index "./DATA/${data}/${data}_base.fvecs" "./DATA/${data}/${data}_${K}.graph" 500 60 60 "./DATA/${data}/${data}.ssg"

done
